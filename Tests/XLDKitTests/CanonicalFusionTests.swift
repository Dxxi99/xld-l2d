import XCTest
import Foundation
@testable import XLDKit

final class CanonicalFusionTests: XCTestCase {

    func testCanonicalFusionMatchesPython() throws {
        let dir = URL(fileURLWithPath: "assets/real/swift_cache")
        let fm = FileManager.default

        func readBinary<T>(_ name: String, type: T.Type) throws -> [T] {
            let url = dir.appendingPathComponent(name)
            guard fm.fileExists(atPath: url.path) else {
                throw XCTSkip("\(name) missing — run cache_binary.py")
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
        let nCons = meta["n_cons"]!
        print("[meta] n_seg=\(nSeg) n_cons=\(nCons)")

        let segZ = try readBinary("seg_z.bin", type: Float.self)
        let consFlat = try readBinary("cons.bin", type: Int32.self)
        XCTAssertEqual(segZ.count, nSeg + 1)
        XCTAssertEqual(consFlat.count, nCons * 3)

        var constraints = [CanonicalFusion.Constraint]()
        constraints.reserveCapacity(nCons)
        for i in 0..<nCons {
            let s = consFlat[i*3]
            let ca = consFlat[i*3 + 1]
            let cb = consFlat[i*3 + 2]
            constraints.append(.init(stem: s, ca: ca, cb: cb))
        }

        let t0 = Date()
        let result = CanonicalFusion.fuse(segZ: segZ, nSeg: nSeg, constraints: constraints)
        let dt = Date().timeIntervalSince(t0)
        print("[swift] fusion took \(String(format: "%.3f", dt))s")
        print("[swift] scc_count=\(result.sccCount)  topo_len=\(result.topoOrder.count)")
        print("[swift] violations=\(result.violations) / \(nCons)")
        XCTAssertEqual(result.violations, 0, "must have 0 violations")

        // Compare to Python expected
        let expectedZ = try readBinary("expected_z_n.bin", type: Float.self)
        let expectedSccLbl = try readBinary("expected_scc_lbl.bin", type: Int32.self)
        XCTAssertEqual(result.z.count, expectedZ.count)
        XCTAssertEqual(result.sccLabel.count, expectedSccLbl.count)

        // z tolerance: fp32 exact-ish
        var maxDiff: Float = 0
        for i in 0..<expectedZ.count {
            let d = abs(result.z[i] - expectedZ[i])
            if d > maxDiff { maxDiff = d }
        }
        print("[cmp] max |z_swift - z_python| = \(maxDiff)")
        // Diagnose scale drift
        let pyMin = expectedZ[1...].min()!
        let pyMax = expectedZ[1...].max()!
        let swMin = result.z[1...].min()!
        let swMax = result.z[1...].max()!
        print("[cmp] python z range [\(pyMin), \(pyMax)]")
        print("[cmp] swift  z range [\(swMin), \(swMax)]")
        // Layer-level agreement (this is what actually matters)
        let nLayers = 7
        func layerAssign(_ z: [Float]) -> [Int] {
            var sorted = z[1...].sorted()
            var edges = [Float]()
            for i in 0...nLayers {
                let q = Float(i) / Float(nLayers)
                let idx = min(sorted.count - 1, Int(q * Float(sorted.count - 1)))
                edges.append(sorted[idx])
            }
            edges[0] -= 1e-6; edges[nLayers] += 1e-6
            var out = [Int](repeating: 255, count: z.count)
            for s in 1..<z.count {
                for l in 0..<nLayers {
                    if z[s] >= edges[l] && z[s] < edges[l+1] { out[s] = l; break }
                }
            }
            return out
        }
        let pyLayers = layerAssign(expectedZ)
        let swLayers = layerAssign(result.z)
        var layerDiff = 0
        for i in 1..<pyLayers.count where pyLayers[i] != swLayers[i] { layerDiff += 1 }
        print("[cmp] layer mismatch (segment-level): \(layerDiff) / \(nSeg)")
        XCTAssertLessThan(layerDiff, nSeg / 100,
            "layer assignment must agree on ≥99% of segments")

        XCTAssertLessThan(maxDiff, 0.006, "z values must match Python within 1 EPS")

        // SCC count may differ by permutation, but count must match
        XCTAssertEqual(result.sccCount, Int(expectedSccLbl.max() ?? -1) + 1,
                       "scc count must match")

        // Violation re-verify with Python z
        var pyViol = 0
        for c in constraints {
            let zs = expectedZ[Int(c.stem)]
            let m = min(expectedZ[Int(c.ca)], expectedZ[Int(c.cb)])
            if zs > m + 1e-6 { pyViol += 1 }
        }
        print("[cmp] python z violations=\(pyViol)")
    }
}
