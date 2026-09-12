import Foundation

public struct FaceBox: Sendable {
    public let x0: Int
    public let y0: Int
    public let x1: Int
    public let y1: Int
    public let score: Float
    public var width: Int { x1 - x0 + 1 }
    public var height: Int { y1 - y0 + 1 }
}

public final class FaceFinder: @unchecked Sendable {
    public let blockSize: Int
    public let topFraction: Float

    public init(blockSize: Int = 32, topFraction: Float = 0.55) {
        self.blockSize = blockSize
        self.topFraction = topFraction
    }

    /// edge 마스크(0/1)를 받아서 얼굴 bbox 추정.
    /// 전략: 상단부에서 edge 밀도 최고 블록 → 주변 블록 확장 → bbox.
    public func find(edgeMask: [UInt8],
                     width: Int, height: Int) -> FaceBox? {
        let bs = blockSize
        let cols = (width + bs - 1) / bs
        let rows = (height + bs - 1) / bs
        if cols < 3 || rows < 3 { return nil }

        var density = [Float](repeating: 0, count: cols * rows)
        for by in 0..<rows {
            for bx in 0..<cols {
                var edgeCount = 0
                var total = 0
                let y0 = by * bs, y1 = min(y0 + bs, height)
                let x0 = bx * bs, x1 = min(x0 + bs, width)
                for y in y0..<y1 {
                    let row = y * width
                    for x in x0..<x1 {
                        total += 1
                        if edgeMask[row + x] == 1 { edgeCount += 1 }
                    }
                }
                density[by * cols + bx] = total > 0 ? Float(edgeCount) / Float(total) : 0
            }
        }

        // 상단부 row 제한
        let topRowLimit = max(1, Int(Float(rows) * topFraction))

        // 최고 밀도 블록 (상단부)
        var bestIdx = -1
        var bestScore: Float = 0
        for by in 0..<topRowLimit {
            for bx in 0..<cols {
                let s = density[by * cols + bx]
                if s > bestScore { bestScore = s; bestIdx = by * cols + bx }
            }
        }
        if bestIdx < 0 { return nil }

        // seed
        let seedBy = bestIdx / cols
        let seedBx = bestIdx % cols

        // 인접 블록 중 밀도 ≥ 0.5 * bestScore 인 것만 flood
        var visited = [UInt8](repeating: 0, count: cols * rows)
        var queue = [(seedBx, seedBy)]
        visited[seedBy * cols + seedBx] = 1
        var minBx = seedBx, maxBx = seedBx
        var minBy = seedBy, maxBy = seedBy
        let threshold = max(0.35, bestScore * 0.5)

        while let (bx, by) = queue.popLast() {
            if bx < minBx { minBx = bx }
            if bx > maxBx { maxBx = bx }
            if by < minBy { minBy = by }
            if by > maxBy { maxBy = by }

            for (dx, dy) in [(1,0),(-1,0),(0,1),(0,-1)] {
                let nx = bx + dx, ny = by + dy
                if nx < 0 || ny < 0 || nx >= cols || ny >= rows { continue }
                if visited[ny * cols + nx] == 1 { continue }
                if density[ny * cols + nx] < threshold { continue }
                visited[ny * cols + nx] = 1
                queue.append((nx, ny))
            }
        }

        let x0 = minBx * bs
        let y0 = minBy * bs
        let x1 = min(width - 1, (maxBx + 1) * bs - 1)
        let y1 = min(height - 1, (maxBy + 1) * bs - 1)

        // sanity check: bbox가 너무 작거나 크면 reject
        let w = x1 - x0 + 1
        let h = y1 - y0 + 1
        if w < bs * 2 || h < bs * 2 { return nil }
        if w > width / 2 || h > height / 2 { return nil }

        return FaceBox(x0: x0, y0: y0, x1: x1, y1: y1, score: bestScore)
    }

    public func densityMap(edgeMask: [UInt8],
                           width: Int, height: Int) -> [Float] {
        let bs = blockSize
        let cols = (width + bs - 1) / bs
        let rows = (height + bs - 1) / bs
        var out = [Float](repeating: 0, count: cols * rows)
        for by in 0..<rows {
            for bx in 0..<cols {
                var edgeCount = 0
                var total = 0
                let y0 = by * bs, y1 = min(y0 + bs, height)
                let x0 = bx * bs, x1 = min(x0 + bs, width)
                for y in y0..<y1 {
                    let row = y * width
                    for x in x0..<x1 {
                        total += 1
                        if edgeMask[row + x] == 1 { edgeCount += 1 }
                    }
                }
                out[by * cols + bx] = total > 0 ? Float(edgeCount) / Float(total) : 0
            }
        }
        return out
    }
}
