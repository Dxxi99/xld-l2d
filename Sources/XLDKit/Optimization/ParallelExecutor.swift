import Foundation

public final class ParallelExecutor: @unchecked Sendable {
    public init() {}

    /// 독립 태스크 병렬 실행. Dependency-aware scheduling은 DependencyGraph가 담당.
    public func run(_ tasks: [() -> Void]) {
        DispatchQueue.concurrentPerform(iterations: tasks.count) { i in
            tasks[i]()
        }
    }
}
