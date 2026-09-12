import Foundation

public struct InpaintStats: Sendable {
    public let tilesRequested: Int
    public let tilesInpainted: Int
    public let pixelsTouched: Int
    public let disocclusionPixels: Int
    public let gpuMs: Double
    public let cacheHits: Int
    public let cacheMisses: Int
}

public struct InpaintOutput: Sendable {
    public let width: Int
    public let height: Int
    public let rgba: [Float]
    public let stats: InpaintStats
}

public final class MinimalInpaintingEngine: @unchecked Sendable {
    public let structure: InpaintingProvider
    public let appearance: InpaintingProvider
    public let tileSize: Int
    public let margin: Int
    public let fusionAlpha: Float

    private let cache: TileCache

    public init(structure: InpaintingProvider = MangaInpaintingProvider(),
                appearance: InpaintingProvider = AnimeBigLaMaProvider(),
                tileSize: Int = 16,
                margin: Int = 4,
                fusionAlpha: Float = 0.5) {
        self.structure = structure
        self.appearance = appearance
        self.tileSize = tileSize
        self.margin = margin
        self.fusionAlpha = fusionAlpha
        self.cache = TileCache()
    }

    public func inpaint(request: InpaintRequest) -> InpaintOutput? {
        let W = request.width, H = request.height
        let total = W * H
        precondition(request.disocclusionMask.count == total)
        precondition(request.contextRGBA.count == total * 4)

        var tileHasDisocclusion = Set<TileCoord>()
        var disPx = 0
        for y in 0..<H {
            for x in 0..<W {
                if request.disocclusionMask[y * W + x] == 1 {
                    disPx += 1
                    tileHasDisocclusion.insert(TileCoord(x: x / tileSize, y: y / tileSize))
                }
            }
        }

        if disPx == 0 {
            return InpaintOutput(width: W, height: H, rgba: request.contextRGBA,
                                 stats: InpaintStats(tilesRequested: 0, tilesInpainted: 0,
                                                     pixelsTouched: 0, disocclusionPixels: 0,
                                                     gpuMs: 0,
                                                     cacheHits: 0, cacheMisses: 0))
        }

        var output = request.contextRGBA
        let sortedTiles = tileHasDisocclusion.sorted { $0.y == $1.y ? $0.x < $1.x : $0.y < $1.y }

        let t0 = Date()
        var cacheHits = 0, cacheMisses = 0
        var tilesInpainted = 0
        var pixelsTouched = 0

        for tile in sortedTiles {
            let x0 = tile.x * tileSize
            let y0 = tile.y * tileSize
            let tw = min(tileSize, W - x0)
            let th = min(tileSize, H - y0)
            if tw <= 0 || th <= 0 { continue }

            // build context with margin
            let inW = tw + margin * 2
            let inH = th + margin * 2
            var context = [Float](repeating: 0, count: inW * inH * 4)
            for iy in 0..<inH {
                let sy = y0 - margin + iy
                for ix in 0..<inW {
                    let sx = x0 - margin + ix
                    let sIdx: Int
                    if sx < 0 || sy < 0 || sx >= W || sy >= H {
                        sIdx = -1
                    } else {
                        sIdx = (sy * W + sx) * 4
                    }
                    let dIdx = (iy * inW + ix) * 4
                    if sIdx >= 0 {
                        for c in 0..<4 { context[dIdx + c] = request.contextRGBA[sIdx + c] }
                    }
                }
            }

            var maskTile = [UInt8](repeating: 0, count: tw * th)
            for ty in 0..<th {
                for tx in 0..<tw {
                    maskTile[ty * tw + tx] = request.disocclusionMask[(y0 + ty) * W + (x0 + tx)]
                }
            }

            // cache key
            let tileBytes = maskTile
            let key = cache.makeKey(tileBytes: tileBytes,
                                    modelID: "\(structure.name)+\(appearance.name)",
                                    modelVersion: "\(structure.version)+\(appearance.version)")
            let cached: Data? = cache.get(key)
            if cached != nil { cacheHits += 1 } else { cacheMisses += 1 }

            let sReq = InpaintTileRequest(tileX: x0, tileY: y0, tileSize: tw,
                                          contextRGBA: context,
                                          maskTile: maskTile,
                                          margin: margin)
            guard let sRes = structure.inpaintTile(sReq),
                  let aRes = appearance.inpaintTile(sReq) else { continue }

            // fuse
            var fused = [Float](repeating: 0, count: tw * th * 4)
            for i in 0..<(tw * th * 4) {
                fused[i] = sRes.rgba[i] * (1 - fusionAlpha) + aRes.rgba[i] * fusionAlpha
            }

            // apply only where mask=1
            for ty in 0..<th {
                for tx in 0..<tw {
                    if maskTile[ty * tw + tx] == 0 { continue }
                    let dIdx = ((y0 + ty) * W + (x0 + tx)) * 4
                    let sIdx = (ty * tw + tx) * 4
                    for c in 0..<4 { output[dIdx + c] = fused[sIdx + c] }
                    pixelsTouched += 1
                }
            }

            cache.put(key, Data([1]))
            tilesInpainted += 1
        }

        let gpuMs = Date().timeIntervalSince(t0) * 1000

        return InpaintOutput(
            width: W, height: H,
            rgba: output,
            stats: InpaintStats(
                tilesRequested: sortedTiles.count,
                tilesInpainted: tilesInpainted,
                pixelsTouched: pixelsTouched,
                disocclusionPixels: disPx,
                gpuMs: gpuMs,
                cacheHits: cacheHits,
                cacheMisses: cacheMisses
            )
        )
    }
}
