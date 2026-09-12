import Foundation

public struct LayerAssignment: Sendable {
    public let componentLayer: [LayerID?]
    public let layerComponent: [LayerID: Int]
    public let layerPixels: [LayerID: Int]
}

public final class LayerClassifier: @unchecked Sendable {
    public init() {}

    public func assign(labeling: ComponentLabeling,
                       width: Int, height: Int) -> LayerAssignment {
        let totalPx = width * height
        let n = labeling.componentSizes.count

        var layerOf: [LayerID?] = Array(repeating: nil, count: n)
        var layerComp: [LayerID: Int] = [:]
        var layerPx:   [LayerID: Int] = [:]

        struct Cand {
            let id: Int
            let size: Int
            let cx: Float, cy: Float
            let bw: Int, bh: Int
            let aspect: Float
        }

        var candidates: [Cand] = []
        for id in 0..<n {
            let s = labeling.componentSizes[id]
            if s == 0 { continue }
            let bb = labeling.componentBBoxes[id]
            let bw = bb.x1 - bb.x0 + 1, bh = bb.y1 - bb.y0 + 1
            let cx = Float(labeling.componentSumX[id]) / Float(s)
            let cy = Float(labeling.componentSumY[id]) / Float(s)
            candidates.append(Cand(id: id, size: s, cx: cx, cy: cy,
                                   bw: bw, bh: bh, aspect: Float(bw) / Float(bh)))
        }

        // 1) background: bbox covers most of image
        if let bg = candidates.first(where: { c in
            let bb = labeling.componentBBoxes[c.id]
            return bb.x0 <= 2 && bb.y0 <= 2 && bb.x1 >= width - 3 && bb.y1 >= height - 3
        }) {
            layerOf[bg.id] = .background
            layerComp[.background] = bg.id
            layerPx[.background] = bg.size
        }

        // 2) face: 중심에 가깝고, 종횡비 0.5~1.5, 면적 5% 이상
        let faceCand = candidates
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
        }

        // 3) hair: face 위쪽, face보다 위에서 시작
        if let face = faceCand {
            let faceBB = labeling.componentBBoxes[face.id]
            let hairCand = candidates
                .filter { c in
                    c.id != layerComp[.background] && c.id != face.id &&
                    c.size > totalPx / 40 &&
                    labeling.componentBBoxes[c.id].y0 < faceBB.y0 + 20
                }
                .max(by: { $0.size < $1.size })
            if let h = hairCand {
                layerOf[h.id] = .hairFront
                layerComp[.hairFront] = h.id
                layerPx[.hairFront] = h.size
            }
        }

        // 4) eyes / mouth: face 내부의 작은 컴포넌트
        if let face = faceCand {
            let faceBB = labeling.componentBBoxes[face.id]
            let innerSmall = candidates.filter { c in
                layerOf[c.id] == nil &&
                c.size >= 30 && c.size <= totalPx / 200 &&
                c.cx > Float(faceBB.x0) && c.cx < Float(faceBB.x1) &&
                c.cy > Float(faceBB.y0) && c.cy < Float(faceBB.y1)
            }
            let byY = innerSmall.sorted { $0.cy < $1.cy }
            let faceMidY = Float(faceBB.y0 + faceBB.y1) / 2

            // 위쪽 절반 = 눈 후보
            let upperHalf = byY.filter { $0.cy < faceMidY }
            let sortedByX = upperHalf.sorted { $0.cx < $1.cx }
            if sortedByX.count >= 2 {
                layerOf[sortedByX[0].id] = .eyeL
                layerOf[sortedByX[1].id] = .eyeR
                layerComp[.eyeL] = sortedByX[0].id
                layerComp[.eyeR] = sortedByX[1].id
                layerPx[.eyeL] = sortedByX[0].size
                layerPx[.eyeR] = sortedByX[1].size
            }

            // 아래쪽 = 입 후보
            let lowerHalf = byY.filter { $0.cy >= faceMidY }
            if let m = lowerHalf.max(by: { $0.size < $1.size }) {
                layerOf[m.id] = .mouth
                layerComp[.mouth] = m.id
                layerPx[.mouth] = m.size
            }
        }

        return LayerAssignment(componentLayer: layerOf,
                               layerComponent: layerComp,
                               layerPixels: layerPx)
    }
}
