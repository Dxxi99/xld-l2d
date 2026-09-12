import Foundation

public struct ErrorTiles: Sendable {
    public let tileSize: Int
    public let grid: [TileCoord]
    public var count: Int { grid.count }
}

public final class ErrorIsolator: @unchecked Sendable {
    public let tileSize: Int
    public let dilationRadius: Int

    public init(tileSize: Int = 16, dilationRadius: Int = 1) {
        self.tileSize = tileSize
        self.dilationRadius = dilationRadius
    }

    public func isolate(errorMask: [UInt8],
                        width: Int, height: Int) -> ErrorTiles {
        var dilated = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                if errorMask[y * width + x] == 0 { continue }
                for dy in -dilationRadius...dilationRadius {
                    for dx in -dilationRadius...dilationRadius {
                        let nx = x + dx, ny = y + dy
                        if nx < 0 || ny < 0 || nx >= width || ny >= height { continue }
                        dilated[ny * width + nx] = 1
                    }
                }
            }
        }
        var set = Set<TileCoord>()
        for y in 0..<height {
            for x in 0..<width {
                if dilated[y * width + x] == 1 {
                    set.insert(TileCoord(x: x / tileSize, y: y / tileSize))
                }
            }
        }
        let sorted = set.sorted { $0.y == $1.y ? $0.x < $1.x : $0.y < $1.y }
        return ErrorTiles(tileSize: tileSize, grid: sorted)
    }
}
