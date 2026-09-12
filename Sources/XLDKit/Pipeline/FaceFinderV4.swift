import Foundation

public struct FaceFinderV4Result: Sendable {
    public let faceBox: (x0: Int, y0: Int, x1: Int, y1: Int)?
    public let score: Float
    public let faceMask: [UInt8]
    public let enclosedCount: Int
    public let hairDilatedCount: Int
}

public final class FaceFinderV4: @unchecked Sendable {
    /// hair mask dilation 반경 (얼굴 둘레를 감쌀 정도)
    public let hairDilation: Int
    /// 최소 얼굴 크기
    public let minFacePx: Int
    /// 얼굴이 이미지의 최대 몇 %까지 가능한가
    public let maxFacePct: Float
    /// 얼굴 중심이 상단 몇 % 이내
    public let topFraction: Float

    public init(hairDilation: Int = 24,
                minFacePx: Int = 2000,
                maxFacePct: Float = 0.05,
                topFraction: Float = 0.55) {
        self.hairDilation = hairDilation
        self.minFacePx = minFacePx
        self.maxFacePct = maxFacePct
        self.topFraction = topFraction
    }

    public func find(image: RasterImage,
                     characterMask: [UInt8],
                     hairMask: [UInt8],
                     edgeMask: [UInt8]) -> FaceFinderV4Result {
        let W = image.width, H = image.height, total = W * H

        // 1) hair dilate
        var hairDil = [UInt8](repeating: 0, count: total)
        let r = hairDilation
        // 효율을 위해 separable dilation
        var tmp = [UInt8](repeating: 0, count: total)
        for y in 0..<H {
            let row = y * W
            for x in 0..<W {
                var v: UInt8 = 0
                let x0 = max(0, x - r)
                let x1 = min(W - 1, x + r)
                for xx in x0...x1 {
                    if hairMask[row + xx] == 1 { v = 1; break }
                }
                tmp[row + x] = v
            }
        }
        for y in 0..<H {
            for x in 0..<W {
                var v: UInt8 = 0
                let y0 = max(0, y - r)
                let y1 = min(H - 1, y + r)
                for yy in y0...y1 {
                    if tmp[yy * W + x] == 1 { v = 1; break }
                }
                hairDil[y * W + x] = v
            }
        }
        var hairDilCount = 0
        for v in hairDil where v == 1 { hairDilCount += 1 }

        // 2) candidate = char AND hairDilated AND NOT hair AND NOT edge
        var cand = [UInt8](repeating: 0, count: total)
        var candCount = 0
        let topLimit = Int(Float(H) * topFraction)
        for y in 0..<topLimit {
            let row = y * W
            for x in 0..<W {
                let i = row + x
                if characterMask[i] == 0 { continue }
                if hairMask[i] == 1 { continue }
                if hairDil[i] == 0 { continue }
                if edgeMask[i] == 1 { continue }
                cand[i] = 1
                candCount += 1
            }
        }

        // 3) closing (dilate → erode) 로 조각 합치기
        //    간단히: dilate 4, erode 4
        let closeR = 4
        var cd = [UInt8](repeating: 0, count: total)
        for y in 0..<H {
            let row = y * W
            for x in 0..<W {
                var v: UInt8 = 0
                let x0 = max(0, x - closeR)
                let x1 = min(W - 1, x + closeR)
                for xx in x0...x1 {
                    if cand[row + xx] == 1 { v = 1; break }
                }
                cd[row + x] = v
            }
        }
        var closed = [UInt8](repeating: 0, count: total)
        for y in 0..<H {
            for x in 0..<W {
                var v: UInt8 = 1
                let y0 = max(0, y - closeR)
                let y1 = min(H - 1, y + closeR)
                for yy in y0...y1 {
                    if cd[yy * W + x] == 0 { v = 0; break }
                }
                closed[y * W + x] = v
            }
        }

        // 4) CC
        let labeling = ConnectedComponents(minSize: minFacePx)
            .label(mask: closed, width: W, height: H)

        // 5) score: 큰 것 + edge density + aspect
        let maxPx = Int(Float(total) * maxFacePct)
        var bestID = -1
        var bestScore: Float = -1
        for id in 0..<labeling.componentSizes.count {
            let s = labeling.componentSizes[id]
            if s < minFacePx || s > maxPx { continue }
            let bb = labeling.componentBBoxes[id]
            let bw = bb.x1 - bb.x0 + 1, bh = bb.y1 - bb.y0 + 1
            let asp = Float(bw) / Float(bh)
            if asp < 0.4 || asp > 2.5 { continue }

            // edge density in bbox
            var ec = 0, tot = 0
            for y in bb.y0...bb.y1 {
                let row = y * W
                for x in bb.x0...bb.x1 {
                    tot += 1
                    if edgeMask[row + x] == 1 { ec += 1 }
                }
            }
            let density = tot > 0 ? Float(ec) / Float(tot) : 0
            let score = Float(s) * (0.5 + density)
            if score > bestScore { bestScore = score; bestID = id }
        }

        var faceMask = [UInt8](repeating: 0, count: total)
        var faceBox: (Int, Int, Int, Int)? = nil
        var enclosedCount = 0
        if bestID >= 0 {
            for i in 0..<total where labeling.componentID[i] == Int32(bestID) {
                faceMask[i] = 1
                enclosedCount += 1
            }
            let bb = labeling.componentBBoxes[bestID]
            faceBox = (bb.x0, bb.y0, bb.x1, bb.y1)
        }

        return FaceFinderV4Result(faceBox: faceBox,
                                  score: bestScore,
                                  faceMask: faceMask,
                                  enclosedCount: enclosedCount,
                                  hairDilatedCount: hairDilCount)
    }
}
