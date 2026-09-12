import Foundation
import simd

public struct RuntimeRig {
    public let basisX: [Float]
    public let basisY: [Float]
    public let rX: Float
    public let rY: Float
    public let width: Int
    public let height: Int

    public var vertexCount: Int { basisX.count }

    public init(basisX: [Float], basisY: [Float],
                rX: Float, rY: Float, width: Int, height: Int) {
        precondition(basisX.count == basisY.count, "basis size mismatch")
        self.basisX = basisX
        self.basisY = basisY
        self.rX = rX
        self.rY = rY
        self.width = width
        self.height = height
    }

    public init(directory: URL) throws {
        let metaURL = directory.appendingPathComponent("meta.txt")
        let metaText = try String(contentsOf: metaURL, encoding: .utf8)
        var n_v = 0
        var rX: Float = 400
        var rY: Float = 250
        var H = 0
        var W = 0
        for line in metaText.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard parts.count >= 2 else { continue }
            switch parts[0] {
            case "n_v": n_v = Int(parts[1]) ?? 0
            case "R_X": rX = Float(parts[1]) ?? rX
            case "R_Y": rY = Float(parts[1]) ?? rY
            case "H":   H   = Int(parts[1]) ?? H
            case "W":   W   = Int(parts[1]) ?? W
            default: break
            }
        }
        let bx = try RuntimeRig.loadFloat32(url: directory.appendingPathComponent("basis_x.bin"))
        let by = try RuntimeRig.loadFloat32(url: directory.appendingPathComponent("basis_y.bin"))
        precondition(bx.count == n_v, "basis_x count != n_v")
        self.init(basisX: bx, basisY: by, rX: rX, rY: rY, width: W, height: H)
    }

    private static func loadFloat32(url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        var out = [Float](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return out
    }

    public func offsets(yawDeg: Float, pitchDeg: Float) -> [SIMD2<Float>] {
        let sx = sin(yawDeg * .pi / 180) * rX
        let sy = sin(pitchDeg * .pi / 180) * rY
        let n = basisX.count
        var out = [SIMD2<Float>](repeating: .zero, count: n)
        basisX.withUnsafeBufferPointer { bx in
            basisY.withUnsafeBufferPointer { by in
                out.withUnsafeMutableBufferPointer { ob in
                    for i in 0..<n {
                        ob[i] = SIMD2(bx[i] * sx, by[i] * sy)
                    }
                }
            }
        }
        return out
    }
}
