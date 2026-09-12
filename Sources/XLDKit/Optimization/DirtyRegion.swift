import Foundation

public final class DirtyRegionManager: @unchecked Sendable {
    private var dirty: PixelMask?
    private let lock = NSLock()

    public init() {}

    public func mark(width: Int, height: Int, x: Int, y: Int) {
        lock.lock(); defer { lock.unlock() }
        if dirty == nil || dirty?.width != width || dirty?.height != height {
            dirty = PixelMask(width: width, height: height, fill: 0)
        }
        dirty?.set(x, y, 1)
    }

    public func snapshot() -> PixelMask? {
        lock.lock(); defer { lock.unlock() }
        return dirty
    }
}
