import Foundation

public final class TaskScheduler: @unchecked Sendable {
    private var queue: [PipelineStage] = []
    private let lock = NSLock()

    public init() {}

    @discardableResult
    public func enqueue(_ stage: PipelineStage) -> PipelineStage {
        lock.lock(); defer { lock.unlock() }
        queue.append(stage)
        return stage
    }

    public func drain() -> [PipelineStage] {
        lock.lock(); defer { lock.unlock() }
        let q = queue; queue.removeAll(keepingCapacity: true); return q
    }
}
