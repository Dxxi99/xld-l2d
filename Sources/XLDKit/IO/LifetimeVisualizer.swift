import Foundation

public enum LifetimeVisualizer {
    public static func writeLifetime(_ r: LifetimeResult, url: URL) throws {
        let total = r.width * r.height
        var rgba = [Float](repeating: 0, count: total * 4)
        for i in 0..<total {
            if r.lifetimeDisocclusion[i] == 1 {
                rgba[i * 4 + 0] = 1.0
                rgba[i * 4 + 1] = 0.15
                rgba[i * 4 + 2] = 0.15
            } else {
                rgba[i * 4 + 0] = 0.20
                rgba[i * 4 + 1] = 0.20
                rgba[i * 4 + 2] = 0.22
            }
            rgba[i * 4 + 3] = 1.0
        }
        let img = RasterImage(width: r.width, height: r.height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }

    public static func writePose(_ pose: PoseResult, width: Int, height: Int, url: URL) throws {
        let total = width * height
        var rgba = [Float](repeating: 0, count: total * 4)
        for i in 0..<total {
            if pose.disocclusionMask[i] == 1 {
                rgba[i * 4 + 0] = 1.0
                rgba[i * 4 + 1] = 0.15
                rgba[i * 4 + 2] = 0.15
            } else {
                rgba[i * 4 + 0] = 0.05
                rgba[i * 4 + 1] = 0.05
                rgba[i * 4 + 2] = 0.05
            }
            rgba[i * 4 + 3] = 1.0
        }
        let img = RasterImage(width: width, height: height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }

    public static func writeLifetimeOverlay(_ r: LifetimeResult,
                                            original: RasterImage,
                                            url: URL) throws {
        let total = r.width * r.height
        var rgba = original.rgba
        for i in 0..<total {
            if r.lifetimeDisocclusion[i] == 1 {
                let r0 = rgba[i * 4 + 0], g0 = rgba[i * 4 + 1], b0 = rgba[i * 4 + 2]
                rgba[i * 4 + 0] = r0 * 0.35 + 1.00 * 0.65
                rgba[i * 4 + 1] = g0 * 0.35 + 0.15 * 0.65
                rgba[i * 4 + 2] = b0 * 0.35 + 0.15 * 0.65
            }
        }
        let img = RasterImage(width: r.width, height: r.height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }
}
