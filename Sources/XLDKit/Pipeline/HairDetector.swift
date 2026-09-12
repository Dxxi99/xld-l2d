import Foundation

public struct HairResult: Sendable {
    public let hairMask: [UInt8]
    public let hairCount: Int
    public let bbox: (x0: Int, y0: Int, x1: Int, y1: Int)?
    public let topFraction: Float
    public let charBBox: (x0: Int, y0: Int, x1: Int, y1: Int)?
}

public final class HairDetector: @unchecked Sendable {
    public let minBlueExcess: Float
    public let minSat: Float
    public let minVal: Float
    public let topFraction: Float

    public init(minBlueExcess: Float = 0.05,
                minSat: Float = 0.20,
                minVal: Float = 0.25,
                topFraction: Float = 0.42) {
        self.minBlueExcess = minBlueExcess
        self.minSat = minSat
        self.minVal = minVal
        self.topFraction = topFraction
    }

    public func detect(image: RasterImage, characterMask: [UInt8]) -> HairResult {
        let W = image.width, H = image.height, total = W * H
        let rgba = image.rgba

        // character bbox
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
        let charBox: (Int, Int, Int, Int)? = (cx1 >= 0) ? (cx0, cy0, cx1, cy1) : nil

        var mask = [UInt8](repeating: 0, count: total)
        var count = 0
        var hx0 = Int.max, hy0 = Int.max, hx1 = -1, hy1 = -1

        if cx1 >= 0 {
            let charH = cy1 - cy0 + 1
            let topLimit = cy0 + Int(Float(charH) * topFraction)
            for y in cy0..<min(topLimit, H) {
                let row = y * W
                for x in cx0...cx1 {
                    let i = row + x
                    if characterMask[i] == 0 { continue }
                    let r = rgba[i*4], g = rgba[i*4+1], b = rgba[i*4+2]
                    let maxC = max(r, max(g, b))
                    let minC = min(r, min(g, b))
                    let s = maxC > 0 ? (maxC - minC) / maxC : 0
                    let v = maxC
                    if b - r < minBlueExcess { continue }
                    if s < minSat { continue }
                    if v < minVal { continue }
                    mask[i] = 1
                    count += 1
                    if x < hx0 { hx0 = x }
                    if x > hx1 { hx1 = x }
                    if y < hy0 { hy0 = y }
                    if y > hy1 { hy1 = y }
                }
            }
        }

        var bbox: (x0: Int, y0: Int, x1: Int, y1: Int)? = nil
        if count > 500 { bbox = (hx0, hy0, hx1, hy1) }

        return HairResult(hairMask: mask, hairCount: count, bbox: bbox,
                          topFraction: topFraction, charBBox: charBox)
    }
}
