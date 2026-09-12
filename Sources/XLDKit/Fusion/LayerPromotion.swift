import Foundation
import Accelerate

/// Convert per-segment z into per-pixel layer assignment via quantile slicing.
/// Mirrors tools/scripts/layer_promotion_v2.py tail.
public struct LayerPromotion {
    public struct Result {
        public let zNormalized: [Float]      // per-segment, [0,1]
        public let layerId: [UInt8]          // per-pixel, 0..N-1 or 255 for bg
        public let layerCounts: [Int]
        public let quantileEdges: [Float]
    }

    /// - Parameters:
    ///   - segZ: length nSeg+1, index 0 unused
    ///   - segLbl: length H*W, int32 per-pixel segment id
    ///   - charMask: length H*W, uint8 (1 = character pixel)
    ///   - H, W: image dims
    ///   - nLayers: number of layers (default 7)
    public static func promote(
        segZ: [Float],
        segLbl: [Int32],
        charMask: [UInt8],
        H: Int, W: Int,
        nLayers: Int = 7
    ) -> Result {
        let nSeg = segZ.count - 1

        // 1) Normalize segZ (index 1...nSeg)
        var zMin: Float =  .greatestFiniteMagnitude
        var zMax: Float = -.greatestFiniteMagnitude
        for s in 1...nSeg {
            let v = segZ[s]
            if v < zMin { zMin = v }
            if v > zMax { zMax = v }
        }
        let range = zMax - zMin
        var zN = segZ
        if range > 0 {
            for s in 1...nSeg {
                zN[s] = (segZ[s] - zMin) / range
            }
        }
        zN[0] = 0

        // 2) Per-pixel z via segLbl lookup
        // Note: seg_lbl.bin is the "nearest segment" per pixel (already EDT-expanded)
        var zPix = [Float](repeating: 0, count: H * W)
        for i in 0..<(H * W) {
            let s = Int(segLbl[i])
            if s >= 0 && s <= nSeg {
                zPix[i] = zN[s]
            }
        }

        // 3) Collect character-pixel z values for quantile
        var zChar = [Float]()
        zChar.reserveCapacity(H * W / 2)
        for i in 0..<(H * W) where charMask[i] != 0 {
            zChar.append(zPix[i])
        }
        zChar.sort()

        // 4) Quantile edges (matches np.quantile linear interpolation)
        var qEdges = [Float](repeating: 0, count: nLayers + 1)
        let n = zChar.count
        for l in 0...nLayers {
            let q = Float(l) / Float(nLayers)
            let pos = q * Float(n - 1)
            let lo = Int(floor(pos))
            let hi = min(lo + 1, n - 1)
            let frac = pos - Float(lo)
            qEdges[l] = zChar[lo] * (1 - frac) + zChar[hi] * frac
        }
        qEdges[0] -= 1e-6
        qEdges[nLayers] += 1e-6

        // 5) Assign layer id per pixel
        var layerId = [UInt8](repeating: 255, count: H * W)
        var counts = [Int](repeating: 0, count: nLayers)
        for i in 0..<(H * W) {
            if charMask[i] == 0 { continue }
            let z = zPix[i]
            for l in 0..<nLayers {
                if z >= qEdges[l] && z < qEdges[l + 1] {
                    layerId[i] = UInt8(l)
                    counts[l] += 1
                    break
                }
            }
        }

        return Result(zNormalized: zN, layerId: layerId,
                      layerCounts: counts, quantileEdges: qEdges)
    }
}
