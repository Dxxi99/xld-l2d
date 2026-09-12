import Foundation

public final class Renderer: @unchecked Sendable {
    private let ctx: MetalContext
    public init(ctx: MetalContext) { self.ctx = ctx }

    public func renderDirty(region: PixelMask?) {
        // Dirty Region만 재렌더링.
    }
}
