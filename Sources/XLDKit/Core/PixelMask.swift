import Foundation

/// 픽셀 단위 마스크. Bool/UInt8/누적 등 목적별로 사용.
public struct PixelMask: Sendable {
    public let width: Int
    public let height: Int
    public private(set) var bits: [UInt8]

    public init(width: Int, height: Int, fill: UInt8 = 0) {
        self.width = width
        self.height = height
        self.bits = [UInt8](repeating: fill, count: width * height)
    }

    @inline(__always)
    public func value(_ x: Int, _ y: Int) -> UInt8 { bits[y * width + x] }

    public mutating func set(_ x: Int, _ y: Int, _ v: UInt8) {
        bits[y * width + x] = v
    }

    public mutating func setAll(_ v: UInt8) {
        for i in 0..<bits.count { bits[i] = v }
    }
}
