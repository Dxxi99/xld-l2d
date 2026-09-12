import XCTest
@testable import XLDKit

final class FusionSmokeTests: XCTestCase {
    func testRegionsSmoke() throws {
        let W = 32, H = 32
        let rgb = [Float](repeating: 0.5, count: W*H*4)
        var depth = [Float](repeating: 0, count: W*H)
        let charMask = [UInt8](repeating: 1, count: W*H)
        for y in 0..<H {
            for x in 0..<W {
                depth[y*W+x] = x < 16 ? 0.2 : 0.8
            }
        }
        var cfg = FusionEngine.Config()
        cfg.regionMinSize = 50
        let eng = FusionEngine(width: W, height: H, config: cfg)
        let r = eng.run(rgb: rgb, depth: depth, charMask: charMask)
        print("regions:", r.regionLabels.prefix(10))
        print("regionZ:", r.regionZ)
        XCTAssertGreaterThan(r.regionZ.count, 1)
        XCTAssertEqual(r.regionZ.count, 3)  // dummy + 2 regions
    }
}
