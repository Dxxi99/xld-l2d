import Foundation

public enum DepthVisualizer {
    /// grayscale depth (밝음=near, 어두움=far)
    public static func writeDepth(_ d: DepthMap, url: URL) throws {
        var rgba = [Float](repeating: 0, count: d.width * d.height * 4)
        for i in 0..<(d.width * d.height) {
            let v = Float(d.depth[i]) / 255.0
            rgba[i*4+0] = v
            rgba[i*4+1] = v
            rgba[i*4+2] = v
            rgba[i*4+3] = 1.0
        }
        try ImageIO.save(image: RasterImage(width: d.width, height: d.height, rgba: rgba), url: url)
    }

    /// turbo-ish 컬러맵 (near=빨강, far=파랑)
    public static func writeDepthColor(_ d: DepthMap, url: URL) throws {
        var rgba = [Float](repeating: 0, count: d.width * d.height * 4)
        for i in 0..<(d.width * d.height) {
            let t = Float(d.depth[i]) / 255.0
            let (r, g, b) = turbo(t)
            rgba[i*4+0] = r
            rgba[i*4+1] = g
            rgba[i*4+2] = b
            rgba[i*4+3] = 1.0
        }
        try ImageIO.save(image: RasterImage(width: d.width, height: d.height, rgba: rgba), url: url)
    }

    public static func writeConfidence(_ d: DepthMap, url: URL) throws {
        var rgba = [Float](repeating: 0, count: d.width * d.height * 4)
        for i in 0..<(d.width * d.height) {
            let v = Float(d.confidence[i]) / 255.0
            rgba[i*4+0] = v
            rgba[i*4+1] = v * 0.3
            rgba[i*4+2] = 1.0 - v
            rgba[i*4+3] = 1.0
        }
        try ImageIO.save(image: RasterImage(width: d.width, height: d.height, rgba: rgba), url: url)
    }

    private static func turbo(_ t: Float) -> (Float, Float, Float) {
        let x = max(0, min(1, t))
        let r = max(0, min(1, 1.5 - abs(4 * x - 3)))
        let g = max(0, min(1, 1.5 - abs(4 * x - 2)))
        let b = max(0, min(1, 1.5 - abs(4 * x - 1)))
        return (r, g, b)
    }
}
