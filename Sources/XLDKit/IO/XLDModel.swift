import Foundation

public struct XLDModel: Sendable {
    public var layers: [LayerID: PixelMask]
    public var restoration: [LayerID: PixelMask]
    public var metadata: [String: String]

    public init() {
        self.layers = [:]
        self.restoration = [:]
        self.metadata = [:]
    }
}
