import Foundation

public enum ComponentVisualizer {
    public static func writeVisualization(_ labeling: ComponentLabeling,
                                          width: Int, height: Int,
                                          url: URL) throws {
        var rgba = [Float](repeating: 0, count: width * height * 4)
        let count = width * height
        for i in 0..<count {
            let id = labeling.componentID[i]
            if id < 0 {
                rgba[i * 4 + 0] = 0.05
                rgba[i * 4 + 1] = 0.05
                rgba[i * 4 + 2] = 0.05
            } else {
                let h = Float(UInt32(bitPattern: id) &* 2654435761) / Float(UInt32.max)
                let (r, g, b) = hsv(h: h, s: 0.75, v: 0.95)
                rgba[i * 4 + 0] = r
                rgba[i * 4 + 1] = g
                rgba[i * 4 + 2] = b
            }
            rgba[i * 4 + 3] = 1.0
        }
        let img = RasterImage(width: width, height: height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }

    public static func writeComponentMask(_ labeling: ComponentLabeling,
                                          component: Int,
                                          width: Int, height: Int,
                                          url: URL) throws {
        var rgba = [Float](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            if labeling.componentID[i] == Int32(component) {
                rgba[i * 4 + 0] = 1.0
                rgba[i * 4 + 1] = 1.0
                rgba[i * 4 + 2] = 1.0
            }
            rgba[i * 4 + 3] = 1.0
        }
        let img = RasterImage(width: width, height: height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }

    private static func hsv(h: Float, s: Float, v: Float) -> (Float, Float, Float) {
        let i = Int(h * 6) % 6
        let f = h * 6 - Float(Int(h * 6))
        let p = v * (1 - s)
        let q = v * (1 - f * s)
        let t = v * (1 - (1 - f) * s)
        switch i {
        case 0: return (v, t, p)
        case 1: return (q, v, p)
        case 2: return (p, v, t)
        case 3: return (p, q, v)
        case 4: return (t, p, v)
        default: return (v, p, q)
        }
    }
}
