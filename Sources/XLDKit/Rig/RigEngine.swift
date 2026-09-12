import Foundation

public struct DeformerKey: Hashable, Sendable {
    public let layer: LayerID
    public let parameter: String
}

public final class RigEngine: @unchecked Sendable {
    public init() {}
    public func bind(layer: LayerID, parameter: String) -> DeformerKey {
        DeformerKey(layer: layer, parameter: parameter)
    }
}
