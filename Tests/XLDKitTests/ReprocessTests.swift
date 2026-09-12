import XCTest
@testable import XLDKit

final class ReprocessTests: XCTestCase {
    func testReprocessFixesIsolatedError() {
        let w = 5, h = 5
        var labels = [PixelLabel](repeating: PixelLabel(classID: 0, confidence: 0.9),
                                  count: w * h)
        labels[2 * w + 2] = PixelLabel(classID: 2, confidence: 0.1)
        var err = [UInt8](repeating: 0, count: w * h)
        err[2 * w + 2] = 1

        let r = ErrorReprocessor(voteRadius: 2)
        let out = r.reprocess(labels: labels, errorMask: err, width: w, height: h)
        XCTAssertEqual(out.pixelsVisited, 1)
        XCTAssertEqual(out.labels[2 * w + 2].classID, 0)
    }
}
