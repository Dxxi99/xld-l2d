import Foundation

/// Inpainting 요청. disocclusion mask는 픽셀 단위.
public struct InpaintRequest: Sendable {
    public let width: Int
    public let height: Int
    public let disocclusionMask: [UInt8]
    public let contextRGBA: [Float]
    public let tileSize: Int
    public let maxPixels: Int

    public init(width: Int, height: Int,
                disocclusionMask: [UInt8],
                contextRGBA: [Float],
                tileSize: Int = 16,
                maxPixels: Int = 1_000_000) {
        self.width = width
        self.height = height
        self.disocclusionMask = disocclusionMask
        self.contextRGBA = contextRGBA
        self.tileSize = tileSize
        self.maxPixels = maxPixels
    }
}

public struct InpaintTileRequest: Sendable {
    public let tileX: Int
    public let tileY: Int
    public let tileSize: Int
    public let contextRGBA: [Float]     // (tileSize+2*margin)^2 * 4
    public let maskTile: [UInt8]        // (tileSize)^2
    public let margin: Int
}

public struct InpaintTileResult: Sendable {
    public let tileX: Int
    public let tileY: Int
    public let tileSize: Int
    public let rgba: [Float]            // (tileSize)^2 * 4
}

public protocol InpaintingProvider: AnyObject, Sendable {
    var name: String { get }
    var version: String { get }
    var kind: InpaintKind { get }

    /// 한 타일의 inpaint. 엔진이 타일링/병렬화/캐싱을 담당.
    func inpaintTile(_ req: InpaintTileRequest) -> InpaintTileResult?
}

public enum InpaintKind: String, Sendable {
    case structure
    case appearance
}

/// MangaInpainting 자리. 선/구조 담당.
public final class MangaInpaintingProvider: InpaintingProvider, @unchecked Sendable {
    public let name = "MangaInpainting"
    public let version = "stub-v0"
    public let kind: InpaintKind = .structure

    public init() {}

    public func inpaintTile(_ req: InpaintTileRequest) -> InpaintTileResult? {
        // Placeholder: 원본 컨텍스트 그대로 반환. 실제 모델이 들어올 자리.
        let margin = req.margin
        let inW = req.tileSize + margin * 2
        var out = [Float](repeating: 0, count: req.tileSize * req.tileSize * 4)
        for ty in 0..<req.tileSize {
            for tx in 0..<req.tileSize {
                let sx = tx + margin
                let sy = ty + margin
                let sIdx = (sy * inW + sx) * 4
                let dIdx = (ty * req.tileSize + tx) * 4
                for c in 0..<4 { out[dIdx + c] = req.contextRGBA[sIdx + c] }
            }
        }
        return InpaintTileResult(tileX: req.tileX, tileY: req.tileY,
                                 tileSize: req.tileSize, rgba: out)
    }
}

/// Anime Big-LaMa 자리. 색/외관 담당.
public final class AnimeBigLaMaProvider: InpaintingProvider, @unchecked Sendable {
    public let name = "Anime-Big-LaMa"
    public let version = "stub-v0"
    public let kind: InpaintKind = .appearance

    public init() {}

    public func inpaintTile(_ req: InpaintTileRequest) -> InpaintTileResult? {
        let margin = req.margin
        let inW = req.tileSize + margin * 2
        var out = [Float](repeating: 0, count: req.tileSize * req.tileSize * 4)
        for ty in 0..<req.tileSize {
            for tx in 0..<req.tileSize {
                let sx = tx + margin
                let sy = ty + margin
                let sIdx = (sy * inW + sx) * 4
                let dIdx = (ty * req.tileSize + tx) * 4
                for c in 0..<4 { out[dIdx + c] = req.contextRGBA[sIdx + c] }
            }
        }
        return InpaintTileResult(tileX: req.tileX, tileY: req.tileY,
                                 tileSize: req.tileSize, rgba: out)
    }
}
