import Foundation
import simd

public struct XLDRuntimeModel {
    public let mesh: XLDMesh
    public let rig: RuntimeRig
    public let vertexZ: [Float]
    public let meta: XLDRuntimeModelIO.Meta
    public let sourceImageURL: URL
    public let width: Int
    public let height: Int

    public init(mesh: XLDMesh, rig: RuntimeRig, vertexZ: [Float],
                meta: XLDRuntimeModelIO.Meta,
                sourceImageURL: URL, width: Int, height: Int) {
        precondition(mesh.vertexCount == rig.vertexCount, "mesh / rig vertex count mismatch")
        precondition(mesh.vertexCount == vertexZ.count, "mesh / z vertex count mismatch")
        self.mesh = mesh
        self.rig = rig
        self.vertexZ = vertexZ
        self.meta = meta
        self.sourceImageURL = sourceImageURL
        self.width = width
        self.height = height
    }

    public static func load(from directory: URL) throws -> XLDRuntimeModel {
        let meta = try XLDRuntimeModelIO.readMeta(directory: directory)
        let mesh = try XLDRuntimeModelIO.loadMesh(directory: directory, meta: meta)
        let rig = RuntimeRig(
            basisX: try XLDRuntimeModelIO.loadFloat32(url: directory.appendingPathComponent("basis_x.bin")),
            basisY: try XLDRuntimeModelIO.loadFloat32(url: directory.appendingPathComponent("basis_y.bin")),
            rX: meta.rX, rY: meta.rY, width: meta.width, height: meta.height
        )
        let zURL = directory.appendingPathComponent("vertices_z.bin")
        let vertexZ: [Float]
        if FileManager.default.fileExists(atPath: zURL.path) {
            vertexZ = try XLDRuntimeModelIO.loadFloat32(url: zURL)
        } else {
            print("[XLDRuntimeModel] vertices_z.bin not found — using 0.5")
            vertexZ = [Float](repeating: 0.5, count: mesh.vertexCount)
        }
        return XLDRuntimeModel(
            mesh: mesh, rig: rig, vertexZ: vertexZ,
            meta: meta,
            sourceImageURL: directory.appendingPathComponent("source.png"),
            width: meta.width, height: meta.height
        )
    }
}
