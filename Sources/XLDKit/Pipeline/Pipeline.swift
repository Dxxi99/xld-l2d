import Foundation

public final class Pipeline {
    public let stages: [PipelineStage] = PipelineStage.allCases
    public let optimizer: OptimizationEngine

    public init(optimizer: OptimizationEngine = OptimizationEngine()) {
        self.optimizer = optimizer
    }

    @discardableResult
    public func run() -> [PipelineStage] {
        for stage in stages {
            _ = optimizer.scheduler.enqueue(stage)
        }
        return stages
    }
}
