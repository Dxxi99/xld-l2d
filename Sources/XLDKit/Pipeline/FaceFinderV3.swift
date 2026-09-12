import Foundation

public struct FaceFinderV3Result: Sendable {
    public let faceBox: (x0: Int, y0: Int, x1: Int, y1: Int)?
    public let score: Float
    public let faceMask: [UInt8]
    public let faceCompID: Int
    public let charBBox: (x0: Int, y0: Int, x1: Int, y1: Int)?
    public let candidateCount: Int
}

public final class FaceFinderV3: @unchecked Sendable {
    public let topFraction: Float
    public let minFacePx: Int
    public let aspectMin: Float
    public let aspectMax: Float
    public let maxFaceWidthFraction: Float

    public init(topFraction: Float = 0.42,
                minFacePx: Int = 500,
                aspectMin: Float = 0.4,
                aspectMax: Float = 1.8,
                maxFaceWidthFraction: Float = 0.40) {
        self.topFraction = topFraction
        self.minFacePx = minFacePx
        self.aspectMin = aspectMin
        self.aspectMax = aspectMax
        self.maxFaceWidthFraction = maxFaceWidthFraction
    }

    public func find(image: RasterImage,
                     characterMask: [UInt8],
                     hairMask: [UInt8],
                     edgeMask: [UInt8]) -> FaceFinderV3Result {
        let W = image.width, H = image.height, total = W * H

        var cx0 = W, cy0 = H, cx1 = -1, cy1 = -1
        for y in 0..<H {
            let row = y * W
            for x in 0..<W where characterMask[row + x] == 1 {
                if x < cx0 { cx0 = x }
                if x > cx1 { cx1 = x }
                if y < cy0 { cy0 = y }
                if y > cy1 { cy1 = y }
            }
        }
        if cx1 < 0 {
            return FaceFinderV3Result(faceBox: nil, score: 0,
                                      faceMask: [UInt8](repeating: 0, count: total),
                                      faceCompID: -1, charBBox: nil, candidateCount: 0)
        }
        let charH = cy1 - cy0 + 1
        let charW = cx1 - cx0 + 1
        let topLimit = cy0 + Int(Float(charH) * topFraction)

        // 후보 마스크: char AND NOT hair, 상단부
        var cand = [UInt8](repeating: 0, count: total)
        for y in cy0..<min(topLimit, H) {
            let row = y * W
            for x in cx0...cx1 {
                let i = row + x
                if characterMask[i] == 0 { continue }
                if hairMask[i] == 1 { continue }
                cand[i] = 1
            }
        }

        let labeling = ConnectedComponents(minSize: minFacePx)
            .label(mask: cand, width: W, height: H)

        var bestID = -1
        var bestScore: Float = -1
        let maxFaceW = Float(charW) * maxFaceWidthFraction

        for id in 0..<labeling.componentSizes.count {
            let s = labeling.componentSizes[id]
            if s < minFacePx { continue }
            let bb = labeling.componentBBoxes[id]
            let bw = bb.x1 - bb.x0 + 1, bh = bb.y1 - bb.y0 + 1
            let asp = Float(bw) / Float(bh)
            if asp < aspectMin || asp > aspectMax { continue }
            if Float(bw) > maxFaceW { continue }

            var ec = 0, tot = 0
            for y in bb.y0...bb.y1 {
                let row = y * W
                for x in bb.x0...bb.x1 {
                    tot += 1
                    if edgeMask[row + x] == 1 { ec += 1 }
                }
            }
            let density = tot > 0 ? Float(ec) / Float(tot) : 0
            let score = density * logf(Float(s) + 1)
            if score > bestScore { bestScore = score; bestID = id }
        }

        if bestID < 0 {
            return FaceFinderV3Result(faceBox: nil, score: 0,
                                      faceMask: [UInt8](repeating: 0, count: total),
                                      faceCompID: -1,
                                      charBBox: (cx0, cy0, cx1, cy1),
                                      candidateCount: labeling.componentCount)
        }

        var faceMask = [UInt8](repeating: 0, count: total)
        for i in 0..<total where labeling.componentID[i] == Int32(bestID) {
            faceMask[i] = 1
        }
        let bb = labeling.componentBBoxes[bestID]
        return FaceFinderV3Result(faceBox: (bb.x0, bb.y0, bb.x1, bb.y1),
                                  score: bestScore,
                                  faceMask: faceMask,
                                  faceCompID: bestID,
                                  charBBox: (cx0, cy0, cx1, cy1),
                                  candidateCount: labeling.componentCount)
    }
}
