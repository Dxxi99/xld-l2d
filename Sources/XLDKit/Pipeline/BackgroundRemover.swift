import Foundation

public struct BackgroundResult: Sendable {
    public let width: Int
    public let height: Int
    public let backgroundMask: [UInt8]     // 1 = background
    public let characterMask: [UInt8]      // 1 = character
    public let backgroundCount: Int
    public let characterCount: Int
}

public final class BackgroundRemover: @unchecked Sendable {
    /// tolerance: 인접 픽셀 색 차이가 이 값 이하면 같은 영역으로 간주
    public let tolerance: Float
    /// 4- or 8-connectivity
    public let diagonal: Bool

    public init(tolerance: Float = 0.10, diagonal: Bool = true) {
        self.tolerance = tolerance
        self.diagonal = diagonal
    }

    public func remove(image: RasterImage) -> BackgroundResult {
        let W = image.width, H = image.height, total = W * H
        let rgba = image.rgba

        // 테두리에서 시작해서 flood fill. edge는 넘지 못함.
        // edge = luma gradient 강한 곳.

        var luma = [Float](repeating: 0, count: total)
        for i in 0..<total {
            luma[i] = 0.299 * rgba[i*4] + 0.587 * rgba[i*4+1] + 0.114 * rgba[i*4+2]
        }
        var isEdge = [UInt8](repeating: 0, count: total)
        let gradTh: Float = 0.10
        for y in 1..<(H-1) {
            for x in 1..<(W-1) {
                let i = y * W + x
                let gx = luma[i+1] - luma[i-1]
                let gy = luma[i+W] - luma[i-W]
                let g = (gx*gx + gy*gy).squareRoot()
                if g > gradTh { isEdge[i] = 1 }
            }
        }

        var bgMask = [UInt8](repeating: 0, count: total)
        var seedColor = [Float](repeating: 0, count: 4)

        // 테두리 픽셀에서 flood fill 시작. 여러 seed 사용.
        // 4 corners + 각 변 중간점.
        let seeds: [(Int, Int)] = [
            (0, 0), (W-1, 0), (0, H-1), (W-1, H-1),
            (W/2, 0), (W/2, H-1), (0, H/2), (W-1, H/2)
        ]

        var queue: [Int] = []
        queue.reserveCapacity(total / 4)

        for (sx, sy) in seeds {
            let i = sy * W + sx
            if bgMask[i] == 1 { continue }
            // seed의 색
            seedColor[0] = rgba[i*4]
            seedColor[1] = rgba[i*4+1]
            seedColor[2] = rgba[i*4+2]
            seedColor[3] = rgba[i*4+3]

            bgMask[i] = 1
            queue.append(i)

            while let cur = queue.popLast() {
                let cx = cur % W
                let cy = cur / W

                for (dx, dy) in [(1,0),(-1,0),(0,1),(0,-1)] + (diagonal ? [(1,1),(1,-1),(-1,1),(-1,-1)] : []) {
                    let nx = cx + dx, ny = cy + dy
                    if nx < 0 || ny < 0 || nx >= W || ny >= H { continue }
                    let ni = ny * W + nx
                    if bgMask[ni] == 1 { continue }
                    if isEdge[ni] == 1 { continue }

                    // seed 색과 얼마나 다른가
                    let dr = rgba[ni*4] - seedColor[0]
                    let dg = rgba[ni*4+1] - seedColor[1]
                    let db = rgba[ni*4+2] - seedColor[2]
                    let dist = (dr*dr + dg*dg + db*db).squareRoot()
                    if dist > tolerance * 1.732 { continue }

                    bgMask[ni] = 1
                    queue.append(ni)
                }
            }
        }

        var charMask = [UInt8](repeating: 1, count: total)
        var bgCount = 0, charCount = 0
        for i in 0..<total {
            if bgMask[i] == 1 { charMask[i] = 0; bgCount += 1 } else { charCount += 1 }
        }

        return BackgroundResult(width: W, height: H,
                                backgroundMask: bgMask,
                                characterMask: charMask,
                                backgroundCount: bgCount,
                                characterCount: charCount)
    }
}
