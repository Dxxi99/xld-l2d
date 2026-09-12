import Foundation

public enum DiagnosticWriter {
    public static func writeErrorMask(_ mask: [UInt8],
                                      width: Int, height: Int,
                                      url: URL) throws {
        var rgba = [Float](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            if mask[i] == 1 {
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

    public static func writeLabels(_ labels: [PixelLabel],
                                   width: Int, height: Int,
                                   url: URL) throws {
        var rgba = [Float](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            let c: Float
            switch labels[i].classID {
            case 0: c = 0.92
            case 1: c = 0.08
            case 2: c = 0.50
            default: c = 0.0
            }
            rgba[i * 4 + 0] = c
            rgba[i * 4 + 1] = c
            rgba[i * 4 + 2] = c
            rgba[i * 4 + 3] = 1.0
        }
        let img = RasterImage(width: width, height: height, rgba: rgba)
        try ImageIO.save(image: img, url: url)
    }
}
