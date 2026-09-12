import Foundation

public struct DepthMap: Sendable {
    public let width: Int
    public let height: Int
    /// 0 = far, 255 = near
    public let depth: [UInt8]
    /// 0 = low confidence, 255 = high
    public let confidence: [UInt8]
    public let providerName: String
    public let providerVersion: String

    public init(width: Int, height: Int,
                depth: [UInt8], confidence: [UInt8],
                providerName: String, providerVersion: String) {
        self.width = width; self.height = height
        self.depth = depth; self.confidence = confidence
        self.providerName = providerName
        self.providerVersion = providerVersion
    }
}

public protocol DepthEstimator: AnyObject, Sendable {
    var name: String { get }
    var version: String { get }
    func estimate(image: RasterImage) -> DepthMap?
}

public final class DepthStubProvider: DepthEstimator, @unchecked Sendable {
    public let name = "DepthStub"
    public let version = "v0"
    public init() {}
    public func estimate(image: RasterImage) -> DepthMap? {
        let W = image.width, H = image.height, total = W * H
        var d = [UInt8](repeating: 0, count: total)
        var c = [UInt8](repeating: 128, count: total)
        for y in 0..<H {
            let row = y * W
            let v = UInt8(255 - (y * 255 / max(1, H - 1)))
            for x in 0..<W { d[row + x] = v }
        }
        return DepthMap(width: W, height: H, depth: d, confidence: c,
                        providerName: name, providerVersion: version)
    }
}

public final class DepthAnythingV2Provider: DepthEstimator, @unchecked Sendable {
    public let name = "DepthAnythingV2"
    public let version = "unloaded"
    public init() {}
    public func estimate(image: RasterImage) -> DepthMap? { nil }
}
