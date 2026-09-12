import Foundation

public struct LayerAssignment: Sendable {
    public let componentLayer: [LayerID?]
    public let layerComponent: [LayerID: Int]
    public let layerPixels: [LayerID: Int]
    public let layerSource: [LayerID: LayerSourceKind]
}

public enum LayerSourceKind: String, Sendable {
    case flatComponent
    case edgeComponent
    case regionComponent
}

public final class LayerClassifier: @unchecked Sendable {
    public init() {}

    public func assign(flatLabeling: ComponentLabeling,
                       edgeLabeling: ComponentLabeling?,
                       width: Int, height: Int) -> LayerAssignment {
        return assign(regionLabeling: flatLabeling,
                      edgeLabeling: edgeLabeling,
                      width: width, height: height)
    }

    public func assign(regionLabeling: ComponentLabeling,
                       edgeLabeling: ComponentLabeling?,
                       width: Int, height: Int) -> LayerAssignment {
        let totalPx = width * height
        let n = regionLabeling.componentSizes.count
        let Wf = Float(width), Hf = Float(height)

        var layerOf: [LayerID?] = Array(repeating: nil, count: n)
        var layerComp: [LayerID: Int] = [:]
        var layerPx:   [LayerID: Int] = [:]
        var layerSrc:  [LayerID: LayerSourceKind] = [:]

        struct Cand {
            let id: Int
            let size: Int
            let cx: Float, cy: Float
            let bw: Int, bh: Int
            let aspect: Float
        }

        func candList(_ labeling: ComponentLabeling) -> [Cand] {
            var out: [Cand] = []
            for id in 0..<labeling.componentSizes.count {
                let s = labeling.componentSizes[id]
                if s == 0 { continue }
                let bb = labeling.componentBBoxes[id]
                let bw = bb.x1 - bb.x0 + 1, bh = bb.y1 - bb.y0 + 1
                let cx = Float(labeling.componentSumX[id]) / Float(s)
                let cy = Float(labeling.componentSumY[id]) / Float(s)
                out.append(Cand(id: id, size: s, cx: cx, cy: cy,
                                bw: bw, bh: bh, aspect: Float(bw) / Float(bh)))
            }
            return out
        }

        let regionCands = candList(regionLabeling)

        // --- background ---
        // touches all 4 edges
        if let bg = regionCands.first(where: { c in
            let bb = regionLabeling.componentBBoxes[c.id]
            return bb.x0 <= 3 && bb.y0 <= 3
                && bb.x1 >= width - 4 && bb.y1 >= height - 4
        }) {
            layerOf[bg.id] = .background
            layerComp[.background] = bg.id
            layerPx[.background] = bg.size
            layerSrc[.background] = .regionComponent
        }

        // --- face ---
        // relaxed: /100 size, wide aspect, loose centroid
        let faceCand = regionCands
            .filter { c in
                c.id != layerComp[.background] &&
                c.size > totalPx / 100 &&
                c.aspect > 0.4 && c.aspect < 2.5 &&
                abs(c.cx - Wf / 2) < Wf * 0.30 &&
                abs(c.cy - Hf / 2) < Hf * 0.35
            }
            .max(by: { $0.size < $1.size })

        if let f = faceCand {
            layerOf[f.id] = .face
            layerComp[.face] = f.id
            layerPx[.face] = f.size
            layerSrc[.face] = .regionComponent
        }

        // --- hair_front ---
        if let face = faceCand {
            let faceBB = regionLabeling.componentBBoxes[face.id]
            let hairCand = regionCands
                .filter { c in
                    c.id != layerComp[.background] && c.id != face.id &&
                    c.size > totalPx / 200 &&
                    regionLabeling.componentBBoxes[c.id].y0 <= faceBB.y0 + 60
                }
                .max(by: { $0.size < $1.size })
            if let h = hairCand {
                layerOf[h.id] = .hairFront
                layerComp[.hairFront] = h.id
                layerPx[.hairFront] = h.size
                layerSrc[.hairFront] = .regionComponent
            }
        }

        // --- eyes: from edge components inside face bbox ---
        if let face = faceCand, let edge = edgeLabeling {
            let faceBB = regionLabeling.componentBBoxes[face.id]
            let faceMidY = Float(faceBB.y0 + faceBB.y1) / 2
            let faceW = Float(faceBB.x1 - faceBB.x0)
            let faceH = Float(faceBB.y1 - faceBB.y0)

            let edgeCands = candList(edge)

            let eyeCands = edgeCands.filter { c in
                c.size >= 40 &&
                c.size <= totalPx / 400 &&
                c.cy < faceMidY && c.cy > Float(faceBB.y0) &&
                c.cx > Float(faceBB.x0) && c.cx < Float(faceBB.x1) &&
                c.aspect > 0.8 && c.aspect < 3.0 &&
                Float(c.bw) < faceW * 0.5 && Float(c.bh) < faceH * 0.35
            }
            let upperSortedByX = eyeCands.sorted { $0.cx < $1.cx }
            if upperSortedByX.count >= 2 {
                let left  = upperSortedByX.first!
                let right = upperSortedByX.last!
                layerComp[.eyeL] = left.id
                layerComp[.eyeR] = right.id
                layerPx[.eyeL] = left.size
                layerPx[.eyeR] = right.size
                layerSrc[.eyeL] = .edgeComponent
                layerSrc[.eyeR] = .edgeComponent
            }

            // --- mouth: wide edge component in lower face ---
            let faceCx = Float(faceBB.x0 + faceBB.x1) / 2
            let mouthCands = edgeCands.filter { c in
                c.size >= 15 &&
                c.size <= totalPx / 400 &&
                c.cy > faceMidY && c.cy < Float(faceBB.y1) &&
                abs(c.cx - faceCx) < faceW * 0.3 &&
                c.aspect > 1.2 &&
                Float(c.bh) < faceH * 0.20
            }
            if let m = mouthCands.max(by: { $0.size < $1.size }) {
                layerComp[.mouth] = m.id
                layerPx[.mouth] = m.size
                layerSrc[.mouth] = .edgeComponent
            }
        }

        return LayerAssignment(componentLayer: layerOf,
                               layerComponent: layerComp,
                               layerPixels: layerPx,
                               layerSource: layerSrc)
    }
}
