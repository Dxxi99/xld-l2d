import Foundation

public enum MoveVisualizer {
    public static func writeDisocclusion(_ r: MoveResult, url: URL) throws {
        let total = r.width * r.height
        var rgba = [Float](repeating: 0, count: total * 4)
        for i in 0..<total {
            let isDis = r.disocclusionMask[i] == 1
            let isNew = r.newCoverageMask[i] == 1
            let isOrig = r.originalMask[i] == 1

            if isDis {
                rgba[i * 4 + 0] = 1.0
                rgba[i * 4 + 1] = 0.15
                rgba[i * 4 + 2] = 0.15
            } else if isNew {
                rgba[i * 4 + 0] = 0.20
                rgba[i * 4 + 1] = 0.90
                rgba[i * 4 + 2] = 0.30
            } else if isOrig {
                rgba[i * 4 + 0] = 0.25
                rgba[i * 4 + 1] = 0.25
                rgba[i * 4 + 2] = 0.30
            } else {
                rgba[i * 4 + 0] = 0.05
                rgba[i * 4 + 1] = 0.05
                rgba[i * 4 + 2] = 0.05
            }
            rgba[i * 4 + 3] = 1.0
        }
        let img = RasterImage(width: r.width, height: r.height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }

    public static func writeOverlay(_ r: MoveResult,
                                    original: RasterImage,
                                    url: URL) throws {
        let total = r.width * r.height
        var rgba = original.rgba
        for i in 0..<total {
            if r.disocclusionMask[i] == 1 {
                rgba[i * 4 + 0] = rgba[i * 4 + 0] * 0.4 + 1.0 * 0.6
                rgba[i * 4 + 1] = rgba[i * 4 + 1] * 0.4 + 0.15 * 0.6
                rgba[i * 4 + 2] = rgba[i * 4 + 2] * 0.4 + 0.15 * 0.6
            } else if r.newCoverageMask[i] == 1 {
                rgba[i * 4 + 0] = rgba[i * 4 + 0] * 0.4 + 0.20 * 0.6
                rgba[i * 4 + 1] = rgba[i * 4 + 1] * 0.4 + 0.90 * 0.6
                rgba[i * 4 + 2] = rgba[i * 4 + 2] * 0.4 + 0.30 * 0.6
            }
        }
        let img = RasterImage(width: r.width, height: r.height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }
}
