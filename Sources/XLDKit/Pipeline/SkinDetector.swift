import Foundation

public final class SkinDetector: @unchecked Sendable {
    public let rMin: Float
    public let rbMin: Float
    public let rbMax: Float
    public let sMax: Float
    public let vMin: Float

    public init(rMin: Float = 0.72,
                rbMin: Float = 0.08,
                rbMax: Float = 0.50,
                sMax: Float = 0.60,
                vMin: Float = 0.70) {
        self.rMin = rMin
        self.rbMin = rbMin
        self.rbMax = rbMax
        self.sMax = sMax
        self.vMin = vMin
    }

    /// 픽셀 단위 피부 마스크.
    public func detect(image: RasterImage) -> [UInt8] {
        let total = image.width * image.height
        var out = [UInt8](repeating: 0, count: total)
        let rgba = image.rgba
        for i in 0..<total {
            let r = rgba[i*4+0]
            let g = rgba[i*4+1]
            let b = rgba[i*4+2]
            let a = rgba[i*4+3]
            if a < 0.5 { continue }
            if r < rMin { continue }
            if !(r > g && g > b) { continue }
            let rb = r - b
            if rb < rbMin || rb > rbMax { continue }
            let maxC = max(r, max(g, b))
            let minC = min(r, min(g, b))
            let v = maxC
            let s = maxC > 0 ? (maxC - minC) / maxC : 0
            if v < vMin { continue }
            if s > sMax { continue }
            out[i] = 1
        }
        return out
    }
}

public struct FaceCandidates: Sendable {
    public let skinMask: [UInt8]
    public let faceBox: (x0: Int, y0: Int, x1: Int, y1: Int)?
    public let faceCompID: Int
    public let skinCompCount: Int
}

public final class SkinFaceFinder: @unchecked Sendable {
    public let minFacePct: Float
    public let maxFacePct: Float
    public let aspectMin: Float
    public let aspectMax: Float
    public let topFraction: Float

    public init(minFacePct: Float = 0.002,
                maxFacePct: Float = 0.15,
                aspectMin: Float = 0.65,
                aspectMax: Float = 1.55,
                topFraction: Float = 0.60) {
        self.minFacePct = minFacePct
        self.maxFacePct = maxFacePct
        self.aspectMin = aspectMin
        self.aspectMax = aspectMax
        self.topFraction = topFraction
    }

    public func find(image: RasterImage, edgeMask: [UInt8]) -> FaceCandidates {
        let W = image.width, H = image.height, total = W * H

        let skin = SkinDetector().detect(image: image)
        let cc = ConnectedComponents(minSize: 100)
        let labeling = cc.label(mask: skin, width: W, height: H)

        // face 후보: 크기 범위 + aspect + 상단부 + edge 밀도
        var bestID = -1
        var bestScore: Float = -1
        var bestBox: (x0: Int, y0: Int, x1: Int, y1: Int)? = nil

        let minPx = Int(Float(total) * minFacePct)
        let maxPx = Int(Float(total) * maxFacePct)

        for id in 0..<labeling.componentSizes.count {
            let s = labeling.componentSizes[id]
            if s < minPx || s > maxPx { continue }
            let bb = labeling.componentBBoxes[id]
            let bw = bb.x1 - bb.x0 + 1, bh = bb.y1 - bb.y0 + 1
            let asp = Float(bw) / Float(bh)
            if asp < aspectMin || asp > aspectMax { continue }
            let cy = Float(bb.y0 + bb.y1) / 2
            if cy > Float(H) * topFraction { continue }

            // edge density inside bbox
            var edgeCount = 0
            var boxTotal = 0
            for y in bb.y0...bb.y1 {
                let row = y * W
                for x in bb.x0...bb.x1 {
                    boxTotal += 1
                    if edgeMask[row + x] == 1 { edgeCount += 1 }
                }
            }
            let density = boxTotal > 0 ? Float(edgeCount) / Float(boxTotal) : 0

            // score = density * log(size)
            let sizeTerm = logf(Float(s) + 1)
            let score = density * sizeTerm

            if score > bestScore {
                bestScore = score
                bestID = id
                bestBox = bb
            }
        }

        return FaceCandidates(skinMask: skin,
                              faceBox: bestBox,
                              faceCompID: bestID,
                              skinCompCount: labeling.componentCount)
    }
}
