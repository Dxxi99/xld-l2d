import XCTest
import Foundation
@testable import XLDKit

final class LayerPromotionTests: XCTestCase {

    func testLayerPromotionMatchesPython() throws {
        let dir = URL(fileURLWithPath: "assets/real/swift_cache")
        let fm = FileManager.default

        func readBinary<T>(_ name: String, type: T.Type) throws -> [T] {
            let url = dir.appendingPathComponent(name)
            guard fm.fileExists(atPath: url.path) else {
                throw XCTSkip("\(name) missing")
            }
            let data = try Data(contentsOf: url)
            let count = data.count / MemoryLayout<T>.size
            return data.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: T.self).prefix(count))
            }
        }

        let metaStr = try String(contentsOf: dir.appendingPathComponent("meta.txt"))
        var meta: [String: Int] = [:]
        for line in metaStr.split(separator: "\n") {
            let parts = line.split(separator: " ")
            if parts.count == 2, let v = Int(parts[1]) { meta[String(parts[0])] = v }
        }
        let nSeg = meta["n_seg"]!
        let H = meta["H"]!, W = meta["W"]!

        let segZ = try readBinary("seg_z.bin", type: Float.self)
        let segLbl = try readBinary("seg_lbl.bin", type: Int32.self)   // raw segment label
        let nearestSeg = try readBinary("nearest_seg.bin", type: Int32.self)
        let charMask = try readBinary("char_mask.bin", type: UInt8.self)
        let expectedLayerId = try readBinary("expected_layer_id.bin", type: UInt8.self)

        XCTAssertEqual(segZ.count, nSeg + 1)
        XCTAssertEqual(segLbl.count, H * W)
        XCTAssertEqual(nearestSeg.count, H * W)
        XCTAssertEqual(charMask.count, H * W)
        XCTAssertEqual(expectedLayerId.count, H * W)

        // 1) First run canonical fusion to get final z
        let consFlat = try readBinary("cons.bin", type: Int32.self)
        let nCons = consFlat.count / 3
        var cons = [CanonicalFusion.Constraint]()
        for i in 0..<nCons {
            cons.append(.init(stem: consFlat[i*3], ca: consFlat[i*3+1], cb: consFlat[i*3+2]))
        }
        let fused = CanonicalFusion.fuse(segZ: segZ, nSeg: nSeg, constraints: cons)

        // 2) Layer promotion using nearest_seg (EDT-expanded) per pixel
        // Swift layer promotion expects per-pixel seg ids; use nearestSeg
        let t0 = Date()
        let promo = LayerPromotion.promote(
            segZ: fused.z,
            segLbl: nearestSeg,
            charMask: charMask,
            H: H, W: W, nLayers: 7
        )
        let dt = Date().timeIntervalSince(t0)
        print("[swift] layer promotion took \(String(format: "%.3f", dt))s")
        print("[swift] layer counts: \(promo.layerCounts)")

        // 3) Compare layer assignments
        var diff = 0
        for i in 0..<(H * W) where promo.layerId[i] != expectedLayerId[i] {
            diff += 1
        }
        let total = H * W
        let pct = 100.0 * Double(diff) / Double(total)
        print("[cmp] layer pixel mismatch: \(diff) / \(total) (\(String(format: "%.3f", pct))%)")

        // Also per-layer diff breakdown
        for l in 0..<7 {
            var d = 0
            for i in 0..<(H*W) {
                if expectedLayerId[i] == UInt8(l) && promo.layerId[i] != UInt8(l) { d += 1 }
            }
            print("  L\(l) mismatch: \(d)")
        }

        // Threshold: <0.1% pixel diff = pass
        XCTAssertLessThan(pct, 0.1, "layer mismatch must be < 0.1% of pixels")
    }
}
