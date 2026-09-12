import Foundation
import simd

public struct XLDMesh {
    public let vertices: [SIMD2<Float>]
    public let triangles: [SIMD3<Int32>]
    public var vertexCount: Int { vertices.count }
    public var triangleCount: Int { triangles.count }
    public init(vertices: [SIMD2<Float>], triangles: [SIMD3<Int32>]) {
        self.vertices = vertices
        self.triangles = triangles
    }
}
