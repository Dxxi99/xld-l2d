import XCTest
@testable import XLDKit

final class PixelUnitTests: XCTestCase {
    func testContextBlockSize() {
        let data = [Float](repeating: 0, count: 4 * 4 * 4)
        let block = ContextBlock(origin: PixelUnit(x: 0, y: 0), width: 4, height: 4, data: data)
        XCTAssertEqual(block.data.count, 64)
    }

    func testPixelMask() {
        var m = PixelMask(width: 4, height: 4, fill: 0)
        m.set(2, 3, 1)
        XCTAssertEqual(m.value(2, 3), 1)
        XCTAssertEqual(m.value(0, 0), 0)
    }
}
