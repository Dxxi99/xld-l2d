import Foundation

public final class OptimizationEngine: Sendable {
    public let scheduler: TaskScheduler
    public let dependencyGraph: DependencyGraph
    public let aiCache: AICache
    public let dirtyRegion: DirtyRegionManager
    public let executor: ParallelExecutor

    public init() {
        self.scheduler       = TaskScheduler()
        self.dependencyGraph = DependencyGraph()
        self.aiCache         = AICache()
        self.dirtyRegion     = DirtyRegionManager()
        self.executor        = ParallelExecutor()
    }
}
