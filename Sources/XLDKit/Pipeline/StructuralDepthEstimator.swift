import Foundation

public final class StructuralDepthEstimator: @unchecked Sendable {
    public let scales: [Int]
    public let characterWeight: Float

    public init(scales: [Int] = [8, 24, 64], characterWeight: Float = 0.35) {
        self.scales = scales
        self.characterWeight = characterWeight
    }

    /// edge mask 기반 pseudo-depth. character mask가 있으면 배경 감쇠.
    public func estimate(image: RasterImage,
                         edgeMask: [UInt8],
                         characterMask: [UInt8]?) -> DepthMap {
        let W = image.width, H = image.height, total = W * H

        // multi-scale box blur of edge mask
        var acc = [Float](repeating: 0, count: total)
        var weightSum: Float = 0
        for (si, r) in scales.enumerated() {
            let w: Float = 1.0 / Float(si + 1)
            weightSum += w
            let blurred = boxBlur(edgeMask, width: W, height: H, radius: r)
            for i in 0..<total { acc[i] += blurred[i] * w }
        }
        for i in 0..<total { acc[i] /= weightSum }

        // character 감쇠: 배경은 depth 낮게 눌러버림
        if let cm = characterMask {
            for i in 0..<total {
                if cm[i] == 0 {
                    acc[i] *= (1.0 - characterWeight)
                }
            }
        }

        // normalize
        var mn: Float = .greatestFiniteMagnitude
        var mx: Float = -.greatestFiniteMagnitude
        for v in acc {
            if v < mn { mn = v }
            if v > mx { mx = v }
        }
        let range = max(1e-6, mx - mn)
        var depth = [UInt8](repeating: 0, count: total)
        var conf = [UInt8](repeating: 160, count: total)
        for i in 0..<total {
            let t = (acc[i] - mn) / range
            depth[i] = UInt8(max(0, min(255, t * 255)))
        }

        return DepthMap(width: W, height: H, depth: depth, confidence: conf,
                        providerName: "StructuralDensity", providerVersion: "v1")
    }

    private func boxBlur(_ src: [UInt8], width: Int, height: Int, radius: Int) -> [Float] {
        let total = width * height
        var tmp = [Float](repeating: 0, count: total)
        // horizontal
        for y in 0..<height {
            let row = y * width
            var sum: Float = 0
            // prime window [0, radius]
            let x1p = min(width - 1, radius)
            for xx in 0...x1p { sum += Float(src[row + xx]) }
            let denom = Float(x1p + 1)
            tmp[row + 0] = sum / denom
            for x in 1..<width {
                if x - 1 - radius >= 0 { sum -= Float(src[row + x - 1 - radius]) }
                if x + radius < width { sum += Float(src[row + x + radius]) }
                let cnt = Float(min(width - 1, x + radius) - max(0, x - radius) + 1)
                tmp[row + x] = sum / max(1, cnt)
            }
        }
        // vertical
        var out = [Float](repeating: 0, count: total)
        for x in 0..<width {
            var sum: Float = 0
            let y1p = min(height - 1, radius)
            for yy in 0...y1p { sum += tmp[yy * width + x] }
            let denom = Float(y1p + 1)
            out[0 * width + x] = sum / denom
            for y in 1..<height {
                if y - 1 - radius >= 0 { sum -= tmp[(y - 1 - radius) * width + x] }
                if y + radius < height { sum += tmp[(y + radius) * width + x] }
                let cnt = Float(min(height - 1, y + radius) - max(0, y - radius) + 1)
                out[y * width + x] = sum / max(1, cnt)
            }
        }
        return out
    }
}
