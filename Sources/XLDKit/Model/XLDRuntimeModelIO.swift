import Foundation
import simd

public enum XLDRuntimeModelIO {

    public struct Meta {
        public let n_v: Int
        public let n_t: Int
        public let rX: Float
        public let rY: Float
        public let height: Int
        public let width: Int
        public let autoFacePivotY: Float
        public let autoHeadZLo: Float
        public let autoHeadZHi: Float
        public let autoLowerYStart: Float
        public let autoLowerYEnd: Float
        public let autoFaceRotScale: Float
        public let autoTuneR: Float
    }

    public static func readMeta(directory: URL) throws -> Meta {
        let text = try String(contentsOf: directory.appendingPathComponent("meta.txt"), encoding: .utf8)
        var n_v = 0
        var n_t = 0
        var H = 0
        var W = 0
        var rX: Float = 400
        var rY: Float = 250
        var autoFacePivotY: Float = 0.32
        var autoHeadZLo: Float = 0.55
        var autoHeadZHi: Float = 0.88
        var autoLowerYStart: Float = 0.55
        var autoLowerYEnd: Float = 0.95
        var autoFaceRotScale: Float = 0.55
        var autoTuneR: Float = 1.8
        for line in text.split(separator: "\n") {
            let p = line.split(separator: " ")
            guard p.count >= 2 else { continue }
            switch p[0] {
            case "n_v": n_v = Int(p[1]) ?? 0
            case "n_t": n_t = Int(p[1]) ?? 0
            case "R_X": rX = Float(p[1]) ?? rX
            case "R_Y": rY = Float(p[1]) ?? rY
            case "auto_face_pivot_y": autoFacePivotY = Float(p[1]) ?? autoFacePivotY
            case "auto_head_z_lo":    autoHeadZLo = Float(p[1]) ?? autoHeadZLo
            case "auto_head_z_hi":    autoHeadZHi = Float(p[1]) ?? autoHeadZHi
            case "auto_lower_y_start": autoLowerYStart = Float(p[1]) ?? autoLowerYStart
            case "auto_lower_y_end":   autoLowerYEnd = Float(p[1]) ?? autoLowerYEnd
            case "auto_face_rot_scale": autoFaceRotScale = Float(p[1]) ?? autoFaceRotScale
            case "auto_tune_r":        autoTuneR = Float(p[1]) ?? autoTuneR
            case "H":   H   = Int(p[1]) ?? H
            case "W":   W   = Int(p[1]) ?? W
            default: break
            }
        }
        return Meta(n_v: n_v, n_t: n_t, rX: rX, rY: rY, height: H, width: W,
                    autoFacePivotY: autoFacePivotY, autoHeadZLo: autoHeadZLo,
                    autoHeadZHi: autoHeadZHi, autoLowerYStart: autoLowerYStart,
                    autoLowerYEnd: autoLowerYEnd, autoFaceRotScale: autoFaceRotScale,
                    autoTuneR: autoTuneR)
    }

    public static func loadMesh(directory: URL, meta: Meta) throws -> XLDMesh {
        let vData = try Data(contentsOf: directory.appendingPathComponent("vertices_xy.bin"))
        precondition(vData.count == meta.n_v * 2 * MemoryLayout<Float>.size, "vertices size mismatch")
        var verts = [SIMD2<Float>](repeating: .zero, count: meta.n_v)
        vData.withUnsafeBytes { raw in
            let f = raw.bindMemory(to: Float.self)
            for i in 0..<meta.n_v {
                verts[i] = SIMD2(f[i*2], f[i*2+1])
            }
        }
        let tData = try Data(contentsOf: directory.appendingPathComponent("triangles.bin"))
        precondition(tData.count == meta.n_t * 3 * MemoryLayout<Int32>.size, "triangles size mismatch")
        var tris = [SIMD3<Int32>](repeating: .zero, count: meta.n_t)
        tData.withUnsafeBytes { raw in
            let p = raw.bindMemory(to: Int32.self)
            for i in 0..<meta.n_t {
                tris[i] = SIMD3(p[i*3], p[i*3+1], p[i*3+2])
            }
        }
        return XLDMesh(vertices: verts, triangles: tris)
    }

    public static func loadFloat32(url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        var out = [Float](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return out
    }
}
