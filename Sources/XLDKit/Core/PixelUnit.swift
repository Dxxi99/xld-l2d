import Foundation

/// Atomic Unit: 하나의 픽셀.
public struct PixelUnit: Hashable, Sendable, Codable {
    public let x: Int
    public let y: Int
    public init(x: Int, y: Int) { self.x = x; self.y = y }
}

/// Flattened pixel index (y * width + x).
public struct PixelIndex: Hashable, Sendable, Codable {
    public let raw: Int
    public init(_ raw: Int) { self.raw = raw }
}
