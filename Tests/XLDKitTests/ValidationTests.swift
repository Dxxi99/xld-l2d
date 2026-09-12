import XCTest
@testable import XLDKit

final class ValidationTests: XCTestCase {
    func testValidatorRejectsLowConfidence() {
        let labels: [PixelLabel] = [
            PixelLabel(classID: 1, confidence: 0.9),
            PixelLabel(classID: 0, confidence: 0.1),
            PixelLabel(classID: 0, confidence: 0.9),
            PixelLabel(classID: 0, confidence: 0.9)
        ]
        let v = PixelValidator()
        let r = v.validate(labels: labels, width: 2, height: 2)
        XCTAssertEqual(r.errorCount, 1)
        XCTAssertEqual(r.acceptedCount, 3)
    }

    func testErrorIsolatorGridSize() {
        var err = [UInt8](repeating: 0, count: 32 * 32)
        err[20 * 32 + 20] = 1
        let iso = ErrorIsolator(tileSize: 16, dilationRadius: 0)
        let res = iso.isolate(errorMask: err, width: 32, height: 32)
        XCTAssertEqual(res.count, 1)
        XCTAssertEqual(res.grid[0].x, 1)
        XCTAssertEqual(res.grid[0].y, 1)
    }

    func testTileExtractorPlan() {
        let ex = TileExtractor(tileSize: 16, overlap: 4)
        let plan = ex.plan(width: 256, height: 256)
        XCTAssertGreaterThan(plan.count, 0)
        XCTAssertEqual(plan.origins[0].x, 0)
        XCTAssertEqual(plan.origins[0].y, 0)
    }

    func testTileCacheHitMiss() {
        let c = TileCache()
        let k = c.makeKey(tileBytes: [1, 2, 3], modelID: "m", modelVersion: "v")
        XCTAssertNil(c.get(k))
        c.put(k, Data([42]))
        XCTAssertNotNil(c.get(k))
        let s = c.snapshotStats()
        XCTAssertEqual(s.stores, 1)
        XCTAssertEqual(s.hits, 1)
        XCTAssertEqual(s.misses, 1)
    }
}
