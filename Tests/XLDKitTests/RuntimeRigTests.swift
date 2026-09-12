import XCTest
@testable import XLDKit

final class RuntimeRigTests: XCTestCase {

    private static func cacheDir() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("XLD-L2D/assets/real/swift_cache/rig_v12")
    }

    func testLoadAndOffsetsMatchPython() throws {
        let dir = RuntimeRigTests.cacheDir()
        let rig = try RuntimeRig(directory: dir)

        XCTAssertGreaterThan(rig.vertexCount, 0)
        print("[meta] n_v=\(rig.vertexCount)  R_X=\(rig.rX)  R_Y=\(rig.rY)  H=\(rig.height) W=\(rig.width)")

        let offs = rig.offsets(yawDeg: 25, pitchDeg: 15)
        let n = offs.count
        var dx = [Float](repeating: 0, count: n)
        var dy = [Float](repeating: 0, count: n)
        for i in 0..<n { dx[i] = offs[i].x; dy[i] = offs[i].y }

        let expX = try RuntimeRigTests.loadFloat32(url: dir.appendingPathComponent("expected_off_x_yaw25.bin"))
        let expY = try RuntimeRigTests.loadFloat32(url: dir.appendingPathComponent("expected_off_y_pitch15.bin"))
        XCTAssertEqual(expX.count, n)
        XCTAssertEqual(expY.count, n)

        var maxDiffX: Float = 0
        var maxDiffY: Float = 0
        for i in 0..<n {
            maxDiffX = max(maxDiffX, abs(dx[i] - expX[i]))
            maxDiffY = max(maxDiffY, abs(dy[i] - expY[i]))
        }
        print("[diff] max |dx| = \(maxDiffX)  max |dy| = \(maxDiffY)")
        XCTAssertLessThan(maxDiffX, 1e-5, "basis_x offset mismatch")
        XCTAssertLessThan(maxDiffY, 1e-5, "basis_y offset mismatch")
    }

    func testPerformance() throws {
        let rig = try RuntimeRig(directory: RuntimeRigTests.cacheDir())
        measure {
            _ = rig.offsets(yawDeg: 25, pitchDeg: 15)
        }
    }

    private static func loadFloat32(url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        var out = [Float](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return out
    }
}
