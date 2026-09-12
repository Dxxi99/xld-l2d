import Foundation

public struct InpaintResult: Sendable {
    public let structure: [Float]   // MangaInpainting
    public let appearance: [Float]  // Anime Big-LaMa
    public let fused: [Float]
}

public final class InpaintingModels: @unchecked Sendable {
    private let runtime: AIRuntime
    public init(runtime: AIRuntime) { self.runtime = runtime }

    public func inpaint(block: ContextBlock) -> InpaintResult {
        let structure  = runtime.infer(model: .mangaInpainting, block: block)
        let appearance = runtime.infer(model: .animeBigLaMa,  block: block)
        let fused = zip(structure, appearance).map { ($0 + $1) * 0.5 }
        return InpaintResult(structure: structure, appearance: appearance, fused: fused)
    }
}
