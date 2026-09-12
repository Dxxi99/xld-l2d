import XCTest
@testable import XLDKit

final class MetalTests: XCTestCase {
    func testMetalValidatorIfAvailable() throws {
        guard let metal = MetalPixelValidator() else {
            throw XCTSkip("Metal unavailable on this machine")
        }
        let w = 8, h = 8
        var labels = [PixelLabel](repeating: PixelLabel(classID: 0, confidence: 0.9),
                                  count: w * h)
        labels[3 * w + 3] = PixelLabel(classID: 2, confidence: 0.1)

        guard let r = metal.validate(labels: labels, width: w, height: h) else {
            XCTFail("validate returned nil")
            return
        }
        XCTAssertGreaterThanOrEqual(r.errorCount, 1)
    }

    func testCPUGPUAgreement() throws {
        guard let metal = MetalPixelValidator() else {
            throw XCTSkip("Metal unavailable")
        }
        let w = 32, h = 32
        var labels = [PixelLabel](repeating: PixelLabel(classID: 0, confidence: 0.9),
                                  count: w * h)
        for i in 0..<(w * h) where i % 7 == 0 {
            labels[i] = PixelLabel(classID: 1, confidence: 0.3)
        }

        let cpu = PixelValidator()
        let rc = cpu.validate(labels: labels, width: w, height: h)
        guard let rg = metal.validate(labels: labels, width: w, height: h) else {
            XCTFail("metal nil"); return
        }
        XCTAssertEqual(rc.errorCount, rg.errorCount)
    }
}
