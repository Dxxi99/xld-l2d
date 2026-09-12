import Foundation

public struct MovementEnvelope: Sendable {
    public struct LayerEnv: Sendable {
        public let layer: LayerID
        public let bboxX0: Int
        public let bboxY0: Int
        public let bboxX1: Int
        public let bboxY1: Int
        public let expandedX0: Int
        public let expandedY0: Int
        public let expandedX1: Int
        public let expandedY1: Int
        public let layerPixels: Int
        public let envelopePixels: Int
        public let disocclusionBudget: Int
    }
    public let budget: Int
    public let layers: [LayerID: LayerEnv]
}

public final class MovementEnvelopeBuilder: @unchecked Sendable {
    public let budget: Int

    public init(budget: Int = 20) { self.budget = budget }

    public func build(layerMasks: [LayerID: [UInt8]],
                      width: Int, height: Int) -> MovementEnvelope {
        var out: [LayerID: MovementEnvelope.LayerEnv] = [:]
        for (layer, mask) in layerMasks {
            var x0 = Int.max, y0 = Int.max, x1 = -1, y1 = -1
            var count = 0
            for y in 0..<height {
                for x in 0..<width {
                    if mask[y * width + x] == 1 {
                        if x < x0 { x0 = x }
                        if y < y0 { y0 = y }
                        if x > x1 { x1 = x }
                        if y > y1 { y1 = y }
                        count += 1
                    }
                }
            }
            if count == 0 { continue }
            let ex0 = max(0, x0 - budget)
            let ey0 = max(0, y0 - budget)
            let ex1 = min(width - 1, x1 + budget)
            let ey1 = min(height - 1, y1 + budget)
            let envPx = (ex1 - ex0 + 1) * (ey1 - ey0 + 1)
            out[layer] = MovementEnvelope.LayerEnv(
                layer: layer,
                bboxX0: x0, bboxY0: y0, bboxX1: x1, bboxY1: y1,
                expandedX0: ex0, expandedY0: ey0,
                expandedX1: ex1, expandedY1: ey1,
                layerPixels: count,
                envelopePixels: envPx,
                disocclusionBudget: max(0, envPx - count)
            )
        }
        return MovementEnvelope(budget: budget, layers: out)
    }
}
