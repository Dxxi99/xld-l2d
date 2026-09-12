import Foundation

public struct MoveResult: Sendable {
    public let layer: LayerID
    public let dx: Int
    public let dy: Int
    public let width: Int
    public let height: Int
    public let originalMask: [UInt8]
    public let movedMask: [UInt8]
    public let disocclusionMask: [UInt8]
    public let newCoverageMask: [UInt8]
    public let originalCount: Int
    public let movedCount: Int
    public let disocclusionCount: Int
    public let newCoverageCount: Int
    public let clippedCount: Int
}

public final class LayerMover: @unchecked Sendable {
    public init() {}

    public func move(layer: LayerID,
                     mask: [UInt8],
                     width: Int, height: Int,
                     dx: Int, dy: Int) -> MoveResult {
        let total = width * height
        var moved = [UInt8](repeating: 0, count: total)
        var movedCount = 0
        var clipped = 0

        for y in 0..<height {
            for x in 0..<width {
                if mask[y * width + x] == 0 { continue }
                let nx = x + dx
                let ny = y + dy
                if nx < 0 || ny < 0 || nx >= width || ny >= height {
                    clipped += 1
                    continue
                }
                moved[ny * width + nx] = 1
                movedCount += 1
            }
        }

        var disocclusion = [UInt8](repeating: 0, count: total)
        var newCoverage   = [UInt8](repeating: 0, count: total)
        var disCount = 0, newCount = 0

        for i in 0..<total {
            let o = mask[i]
            let m = moved[i]
            if o == 1 && m == 0 {
                disocclusion[i] = 1
                disCount += 1
            }
            if m == 1 && o == 0 {
                newCoverage[i] = 1
                newCount += 1
            }
        }

        var origCount = 0
        for v in mask where v == 1 { origCount += 1 }

        return MoveResult(layer: layer, dx: dx, dy: dy,
                          width: width, height: height,
                          originalMask: mask,
                          movedMask: moved,
                          disocclusionMask: disocclusion,
                          newCoverageMask: newCoverage,
                          originalCount: origCount,
                          movedCount: movedCount,
                          disocclusionCount: disCount,
                          newCoverageCount: newCount,
                          clippedCount: clipped)
    }
}
