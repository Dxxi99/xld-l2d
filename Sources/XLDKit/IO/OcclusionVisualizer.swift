import Foundation

public enum OcclusionVisualizer {
    public static func writeZones(_ graph: OcclusionGraph, url: URL) throws {
        let palette: [(Float, Float, Float)] = [
            (0.18, 0.18, 0.20),
            (0.20, 0.60, 0.90),
            (0.60, 0.40, 0.90),
            (0.90, 0.70, 0.30),
            (0.30, 0.70, 0.50),
            (0.90, 0.50, 0.50),
            (0.95, 0.55, 0.55),
            (1.00, 0.90, 0.70),
            (0.90, 0.30, 0.30),
            (0.10, 0.10, 0.35),
            (0.10, 0.10, 0.35),
            (0.55, 0.35, 0.15),
            (0.55, 0.35, 0.15),
            (0.15, 0.08, 0.05),
            (0.25, 0.15, 0.10),
        ]

        let total = graph.width * graph.height
        var rgba = [Float](repeating: 0, count: total * 4)
        for i in 0..<total {
            let r = Int(graph.pixelOwnerRank[i])
            if r < 0 {
                rgba[i * 4 + 0] = 0.05
                rgba[i * 4 + 1] = 0.05
                rgba[i * 4 + 2] = 0.05
            } else {
                let c = palette[r % palette.count]
                rgba[i * 4 + 0] = c.0
                rgba[i * 4 + 1] = c.1
                rgba[i * 4 + 2] = c.2
            }
            rgba[i * 4 + 3] = 1.0
        }
        let img = RasterImage(width: graph.width, height: graph.height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }
}
