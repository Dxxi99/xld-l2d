import Foundation

public final class ErrorReprocessor: @unchecked Sendable {
    public struct Result: Sendable {
        public let labels: [PixelLabel]
        public let pixelsVisited: Int
        public let pixelsChanged: Int
        public let pixelsFixed: Int
    }

    public let voteRadius: Int

    public init(voteRadius: Int = 2) {
        self.voteRadius = voteRadius
    }

    public func reprocess(labels: [PixelLabel],
                          errorMask: [UInt8],
                          width: Int,
                          height: Int) -> Result {
        var out = labels
        var visited = 0
        var changed = 0
        var fixed = 0

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if errorMask[idx] == 0 { continue }
                visited += 1

                var votes = [Int](repeating: 0, count: 3)
                var total = 0
                for dy in -voteRadius...voteRadius {
                    for dx in -voteRadius...voteRadius {
                        if dx == 0 && dy == 0 { continue }
                        let nx = x + dx, ny = y + dy
                        if nx < 0 || ny < 0 || nx >= width || ny >= height { continue }
                        let nIdx = ny * width + nx
                        if errorMask[nIdx] == 1 { continue }
                        votes[Int(labels[nIdx].classID)] += 1
                        total += 1
                    }
                }

                if total == 0 { continue }

                var best = 0
                for i in 1..<votes.count where votes[i] > votes[best] { best = i }

                let oldClass = labels[idx].classID
                let newClass = UInt8(best)
                let newLabel = PixelLabel(classID: newClass, confidence: 0.80)
                out[idx] = newLabel

                if newClass != oldClass { changed += 1; fixed += 1 }
                else { fixed += 1 }
            }
        }

        return Result(labels: out,
                      pixelsVisited: visited,
                      pixelsChanged: changed,
                      pixelsFixed: fixed)
    }
}
