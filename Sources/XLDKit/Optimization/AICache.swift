import Foundation

/// Tile 단위 캐시 + 픽셀 단위 유효성.
public final class AICache: @unchecked Sendable {
    public struct Key: Hashable, Sendable {
        public let tileHash: UInt64
        public let modelID: String
        public let modelVersion: String
        public let paramsHash: UInt64
        public let pipelineVersion: String
    }

    private var store: [Key: Data] = [:]
    private let lock = NSLock()

    public init() {}

    public func get(_ key: Key) -> Data? {
        lock.lock(); defer { lock.unlock() }
        return store[key]
    }

    public func put(_ key: Key, _ value: Data) {
        lock.lock(); defer { lock.unlock() }
        store[key] = value
    }
}
