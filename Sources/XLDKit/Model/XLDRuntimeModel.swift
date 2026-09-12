import Foundation
import simd

public struct XLDRuntimeModel {
    public let mesh: XLDMesh
    public let rig: RuntimeRig
    public let sourceImageURL: URL
    public let width: Int
    public let height: Int

    public init(mesh: XLDMesh, rig: RuntimeRig, sourceImageURL: URL, width: Int, height: Int) {
        precondition(mesh.vertexCount == rig.vertexCount, "mesh / rig vertex count mismatch")
        self.mesh = mesh
        self.rig = rig
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
        return XLDRuntimeModel(
            mesh: mesh, rig: rig,
            sourceImageURL: directory.appendingPathComponent("source.png"),
            width: meta.width, height: meta.height
        )
    }
}
