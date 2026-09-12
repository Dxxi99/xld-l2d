import Foundation

public struct DownsampledMask: Sendable {
    public let width: Int
    public let height: Int
    public let factor: Int
    public let mask: [UInt8]
}

public enum Downsampler {
    public static func downsampleMajority(mask: [UInt8],
                                          width: Int, height: Int,
                                          factor: Int) -> DownsampledMask {
        let nw = max(1, width / factor)
        let nh = max(1, height / factor)
        var out = [UInt8](repeating: 0, count: nw * nh)
        for ny in 0..<nh {
            let y0 = ny * factor
            let y1 = min(y0 + factor, height)
            for nx in 0..<nw {
                let x0 = nx * factor
                let x1 = min(x0 + factor, width)
                var oneCount = 0
                var total = 0
                for y in y0..<y1 {
                    let row = y * width
                    for x in x0..<x1 {
                        total += 1
                        if mask[row + x] == 1 { oneCount += 1 }
                    }
                }
                if total > 0 && oneCount * 2 >= total { out[ny * nw + nx] = 1 }
            }
        }
        return DownsampledMask(width: nw, height: nh, factor: factor, mask: out)
    }
}
