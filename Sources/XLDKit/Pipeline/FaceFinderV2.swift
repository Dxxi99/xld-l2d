import Foundation

public struct FaceFinderV2Result: Sendable {
    public let faceBox: (x0: Int, y0: Int, x1: Int, y1: Int)?
    public let score: Float
    public let hairBBox: (x0: Int, y0: Int, x1: Int, y1: Int)?
}

public final class FaceFinderV2: @unchecked Sendable {
    public let blockSize: Int

    public init(blockSize: Int = 32) {
        self.blockSize = blockSize
    }

    /// hair 검출 결과와 edge 밀도로 얼굴 bbox 찾기.
    /// hair bbox의 하단 근처에서 시작해서 edge 밀도 최고 영역을 얼굴로.
    public func find(characterMask: [UInt8],
                     hairMask: [UInt8],
                     hairBBox: (x0: Int, y0: Int, x1: Int, y1: Int)?,
                     edgeMask: [UInt8],
                     width: Int, height: Int) -> FaceFinderV2Result {
        let bs = blockSize
        let cols = (width + bs - 1) / bs
        let rows = (height + bs - 1) / bs

        // edge density map
        var density = [Float](repeating: 0, count: cols * rows)
        var charFrac = [Float](repeating: 0, count: cols * rows)
        var hairFrac = [Float](repeating: 0, count: cols * rows)

        for by in 0..<rows {
            for bx in 0..<cols {
                var edgeCount = 0, charCount = 0, hairCount = 0, total = 0
                let y0 = by * bs, y1 = min(y0 + bs, height)
                let x0 = bx * bs, x1 = min(x0 + bs, width)
                for y in y0..<y1 {
                    let row = y * width
                    for x in x0..<x1 {
                        total += 1
                        let i = row + x
                        if edgeMask[i] == 1 { edgeCount += 1 }
                        if characterMask[i] == 1 { charCount += 1 }
                        if hairMask[i] == 1 { hairCount += 1 }
                    }
                }
                if total > 0 {
                    density[by * cols + bx] = Float(edgeCount) / Float(total)
                    charFrac[by * cols + bx] = Float(charCount) / Float(total)
                    hairFrac[by * cols + bx] = Float(hairCount) / Float(total)
                }
            }
        }

        // 얼굴 후보 조건:
        // - 상단 절반
        // - 캐릭터 비율 높음
        // - hair 비율 낮음 (얼굴 자체는 hair 아님)
        // - edge density 중상
        var bestScore: Float = -1
        var bestIdx = -1

        for by in 0..<rows {
            for bx in 0..<cols {
                let i = by * cols + bx
                if charFrac[i] < 0.5 { continue }
                if hairFrac[i] > 0.5 { continue }
                if density[i] < 0.05 { continue }

                // 상단 가중치: 얼굴은 상단에
                let yPct = Float(by) / Float(rows)
                if yPct > 0.55 { continue }
                let topBonus = (1.0 - yPct) * 0.3

                // hair bbox 하단 근처 보너스
                var hairBonus: Float = 0
                if let hb = hairBBox {
                    let hbBottomBlock = hb.y1 / bs
                    let dist = abs(by - hbBottomBlock)
                    if dist < 4 { hairBonus = 0.2 * (1.0 - Float(dist) / 4.0) }
                }

                let score = density[i] + topBonus + hairBonus
                if score > bestScore { bestScore = score; bestIdx = i }
            }
        }

        guard bestIdx >= 0 else {
            return FaceFinderV2Result(faceBox: nil, score: 0, hairBBox: hairBBox)
        }

        // seed 주변으로 bbox 확장 (조건 만족하는 블록만)
        let seedBy = bestIdx / cols
        let seedBx = bestIdx % cols

        var visited = [UInt8](repeating: 0, count: cols * rows)
        var queue = [(seedBx, seedBy)]
        visited[seedBy * cols + seedBx] = 1
        var minBx = seedBx, maxBx = seedBx
        var minBy = seedBy, maxBy = seedBy

        let expandThreshold: Float = 0.4   // charFrac 최소값
        let hairLimit: Float = 0.5

        while let (bx, by) = queue.popLast() {
            if bx < minBx { minBx = bx }
            if bx > maxBx { maxBx = bx }
            if by < minBy { minBy = by }
            if by > maxBy { maxBy = by }

            for (dx, dy) in [(1,0),(-1,0),(0,1),(0,-1)] {
                let nx = bx + dx, ny = by + dy
                if nx < 0 || ny < 0 || nx >= cols || ny >= rows { continue }
                let ni = ny * cols + nx
                if visited[ni] == 1 { continue }
                if charFrac[ni] < expandThreshold { continue }
                if hairFrac[ni] > hairLimit { continue }
                visited[ni] = 1
                queue.append((nx, ny))
            }
        }

        let x0 = minBx * bs
        let y0 = minBy * bs
        let x1 = min(width - 1, (maxBx + 1) * bs - 1)
        let y1 = min(height - 1, (maxBy + 1) * bs - 1)

        let w = x1 - x0 + 1, h = y1 - y0 + 1
        if w < bs * 2 || h < bs * 2 { return FaceFinderV2Result(faceBox: nil, score: bestScore, hairBBox: hairBBox) }
        if w > width / 2 || h > height / 2 { return FaceFinderV2Result(faceBox: nil, score: bestScore, hairBBox: hairBBox) }

        return FaceFinderV2Result(faceBox: (x0, y0, x1, y1),
                                  score: bestScore,
                                  hairBBox: hairBBox)
    }
}
