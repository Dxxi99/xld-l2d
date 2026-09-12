import Foundation

public final class TileCache: @unchecked Sendable {
    public struct Stats: Sendable {
        public var hits: Int = 0
        public var misses: Int = 0
        public var stores: Int = 0
    }

    private var store: [UInt64: Data] = [:]
    private var stats = Stats()
    private let lock = NSLock()

    public init() {}

    public func makeKey(tileBytes: [UInt8],
                        modelID: String,
                        modelVersion: String) -> UInt64 {
        var h: UInt64 = 1469598103934665603
        for b in tileBytes  { h ^= UInt64(b); h = h &* 1099511628211 }
        for b in modelID.utf8     { h ^= UInt64(b); h = h &* 1099511628211 }
        for b in modelVersion.utf8 { h ^= UInt64(b); h = h &* 1099511628211 }
        return h
    }

    public func get(_ k: UInt64) -> Data? {
        lock.lock(); defer { lock.unlock() }
        if let v = store[k] { stats.hits += 1; return v }
        stats.misses += 1
        return nil
    }

    public func put(_ k: UInt64, _ v: Data) {
        lock.lock(); defer { lock.unlock() }
        store[k] = v
        stats.stores += 1
    }

    public func snapshotStats() -> Stats {
        lock.lock(); defer { lock.unlock() }
        return stats
    }
}
