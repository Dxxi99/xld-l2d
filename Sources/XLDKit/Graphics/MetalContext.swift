import Foundation

public final class MetalContext: @unchecked Sendable {
    public init() {}
    /// 실제 구현: MTLDevice, MTLCommandQueue, library 로드.
    public func makeLibraryPath() -> String {
        return "Shaders/PerPixel.metal"
    }
}
