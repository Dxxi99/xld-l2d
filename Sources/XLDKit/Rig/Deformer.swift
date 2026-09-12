import Foundation

public struct Deformer: Sendable {
    public let key: DeformerKey
    public let amount: Float
    public init(key: DeformerKey, amount: Float) {
        self.key = key
        self.amount = amount
    }
}
