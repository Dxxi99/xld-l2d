import Foundation

public struct TileCoord: Hashable, Sendable {
    public let x: Int
    public let y: Int
    public init(x: Int, y: Int) { self.x = x; self.y = y }
}

public struct ContextPlan: Sendable {
    public let tileSize: Int
    public let overlap: Int
    public let width: Int
    public let height: Int
    public let origins: [TileCoord]
    public var count: Int { origins.count }
}

public struct TileExtractor: Sendable {
    public let tileSize: Int
    public let overlap: Int

    public init(tileSize: Int = 16, overlap: Int = 4) {
        self.tileSize = tileSize
        self.overlap = overlap
    }

    public func plan(width: Int, height: Int) -> ContextPlan {
        var origins: [TileCoord] = []
        let step = max(1, tileSize - overlap)
        var y = 0
        while y < height {
            var x = 0
            while x < width {
                origins.append(TileCoord(x: x, y: y))
                if x + tileSize >= width { break }
                x += step
            }
            if y + tileSize >= height { break }
            y += step
        }
        return ContextPlan(tileSize: tileSize, overlap: overlap,
                           width: width, height: height, origins: origins)
    }
}
