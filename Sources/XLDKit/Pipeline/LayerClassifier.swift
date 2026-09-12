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
}

public final class LayerClassifier: @unchecked Sendable {
    public init() {}

    public func assign(flatLabeling: ComponentLabeling,
                       edgeLabeling: ComponentLabeling?,
                       width: Int, height: Int) -> LayerAssignment {
        let totalPx = width * height
        let n = flatLabeling.componentSizes.count

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

        let flatCands = candList(flatLabeling)

        // background
        if let bg = flatCands.first(where: { c in
            let bb = flatLabeling.componentBBoxes[c.id]
            return bb.x0 <= 2 && bb.y0 <= 2 && bb.x1 >= width - 3 && bb.y1 >= height - 3
        }) {
            layerOf[bg.id] = .background
            layerComp[.background] = bg.id
            layerPx[.background] = bg.size
            layerSrc[.background] = .flatComponent
        }

        // face
        let faceCand = flatCands
            .filter { c in
                c.id != layerComp[.background] &&
                c.size > totalPx / 20 &&
                c.aspect > 0.5 && c.aspect < 1.6 &&
                abs(c.cx - Float(width) / 2)  < Float(width) * 0.20 &&
                abs(c.cy - Float(height) / 2) < Float(height) * 0.25
            }
            .max(by: { $0.size < $1.size })

        if let f = faceCand {
            layerOf[f.id] = .face
            layerComp[.face] = f.id
            layerPx[.face] = f.size
            layerSrc[.face] = .flatComponent
        }

        // hair_front
        if let face = faceCand {
            let faceBB = flatLabeling.componentBBoxes[face.id]
            let hairCand = flatCands
                .filter { c in
                    c.id != layerComp[.background] && c.id != face.id &&
                    c.size > totalPx / 40 &&
                    flatLabeling.componentBBoxes[c.id].y0 < faceBB.y0 + 20
                }
                .max(by: { $0.size < $1.size })
            if let h = hairCand {
                layerOf[h.id] = .hairFront
                layerComp[.hairFront] = h.id
                layerPx[.hairFront] = h.size
                layerSrc[.hairFront] = .flatComponent
            }
        }

        // eyes from flat components
        if let face = faceCand {
            let faceBB = flatLabeling.componentBBoxes[face.id]
            let innerSmall = flatCands.filter { c in
                layerOf[c.id] == nil &&
                c.size >= 30 && c.size <= totalPx / 200 &&
                c.cx > Float(faceBB.x0) && c.cx < Float(faceBB.x1) &&
                c.cy > Float(faceBB.y0) && c.cy < Float(faceBB.y1)
            }
            let faceMidY = Float(faceBB.y0 + faceBB.y1) / 2
            let upper = innerSmall.filter { $0.cy < faceMidY }.sorted { $0.cx < $1.cx }
            if upper.count >= 2 {
                layerOf[upper[0].id] = .eyeL
                layerOf[upper[1].id] = .eyeR
                layerComp[.eyeL] = upper[0].id
                layerComp[.eyeR] = upper[1].id
                layerPx[.eyeL] = upper[0].size
                layerPx[.eyeR] = upper[1].size
                layerSrc[.eyeL] = .flatComponent
                layerSrc[.eyeR] = .flatComponent
            }
        }

        // mouth from edge components
        if let face = faceCand, let edge = edgeLabeling {
            let faceBB = flatLabeling.componentBBoxes[face.id]
            let faceMidY = Float(faceBB.y0 + faceBB.y1) / 2
            let faceCx   = Float(faceBB.x0 + faceBB.x1) / 2

            let edgeCands = candList(edge)

            let mouthCandidates = edgeCands.filter { c in
                let bb = edge.componentBBoxes[c.id]
                let insideFaceX = bb.x0 >= faceBB.x0 && bb.x1 <= faceBB.x1
                let insideFaceY = bb.y0 >= faceBB.y0 && bb.y1 <= faceBB.y1
                let inLowerHalf = c.cy > faceMidY
                let wide = c.aspect > 1.5
                let centered = abs(c.cx - faceCx) < Float(faceBB.x1 - faceBB.x0) * 0.25
                let sizeOK = c.size >= 20 && c.size <= totalPx / 100
                return insideFaceX && insideFaceY && inLowerHalf && wide && centered && sizeOK
            }

            if let m = mouthCandidates.max(by: { $0.size < $1.size }) {
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
