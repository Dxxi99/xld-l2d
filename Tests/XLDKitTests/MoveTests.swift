import XCTest
@testable import XLDKit

final class MoveTests: XCTestCase {
    func testMoveRightUncoversLeft() {
        let w = 10, h = 10
        var mask = [UInt8](repeating: 0, count: w * h)
        // 3x3 block at (2,4)
        for y in 4..<7 {
            for x in 2..<5 { mask[y * w + x] = 1 }
        }
        let mover = LayerMover()
        let r = mover.move(layer: .face, mask: mask, width: w, height: h, dx: 3, dy: 0)

        XCTAssertEqual(r.originalCount, 9)
        XCTAssertEqual(r.movedCount, 9)
        XCTAssertEqual(r.disocclusionCount, 9)   // completely disjoint after moving 3px
        XCTAssertEqual(r.newCoverageCount, 9)
        XCTAssertEqual(r.clippedCount, 0)
    }

    func testMovePartialOverlap() {
        let w = 10, h = 10
        var mask = [UInt8](repeating: 0, count: w * h)
        for y in 4..<7 {
            for x in 2..<5 { mask[y * w + x] = 1 }
        }
        let mover = LayerMover()
        let r = mover.move(layer: .face, mask: mask, width: w, height: h, dx: 1, dy: 0)

        XCTAssertEqual(r.originalCount, 9)
        XCTAssertEqual(r.movedCount, 9)
        // overlap of 2 columns (2 pixels × 3 rows) = 6
        XCTAssertEqual(r.disocclusionCount, 3)
        XCTAssertEqual(r.newCoverageCount, 3)
    }

    func testMoveClipsAtBoundary() {
        let w = 5, h = 5
        var mask = [UInt8](repeating: 0, count: w * h)
        mask[0] = 1
        mask[1] = 1
        let mover = LayerMover()
        let r = mover.move(layer: .face, mask: mask, width: w, height: h, dx: -1, dy: 0)

        XCTAssertEqual(r.clippedCount, 1)
        XCTAssertEqual(r.movedCount, 1)
    }
}
