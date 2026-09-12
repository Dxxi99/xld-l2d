import Foundation

/// Inference Unit: AI가 실제로 보는 문맥 블록.
/// 데이터는 RGBA interleaved, row-major.
public struct ContextBlock: Sendable {
    public let origin: PixelUnit
    public let width: Int
    public let height: Int
    public let stride: Int
    public let data: [Float]

    public init(origin: PixelUnit, width: Int, height: Int, data: [Float]) {
        precondition(data.count == width * height * 4, "RGBA size mismatch")
        self.origin = origin
        self.width = width
        self.height = height
        self.stride = width * 4
        self.data = data
    }
}
