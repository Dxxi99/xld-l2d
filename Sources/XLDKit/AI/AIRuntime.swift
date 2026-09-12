import Foundation

public enum AIModelID: String, Sendable {
    case segmentation
    case mangaInpainting
    case animeBigLaMa
}

public final class AIRuntime: @unchecked Sendable {
    public init() {}

    /// 실제 구현에서는 CoreML / MPSGraph / Metal Performance Shaders 사용.
    public func infer(model: AIModelID, block: ContextBlock) -> [Float] {
        // Placeholder: pass-through
        return block.data
    }
}
