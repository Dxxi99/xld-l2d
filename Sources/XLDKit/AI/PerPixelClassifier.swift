import Foundation

public struct PixelLabel: Sendable {
    public let classID: UInt8
    public let confidence: Float
    public init(classID: UInt8, confidence: Float) {
        self.classID = classID
        self.confidence = confidence
    }
}

public final class PerPixelClassifier: @unchecked Sendable {
    public let edgeThreshold: Float
    public let textureThreshold: Float

    public init(edgeThreshold: Float = 0.08, textureThreshold: Float = 0.02) {
        self.edgeThreshold = edgeThreshold
        self.textureThreshold = textureThreshold
    }

    public func classify(_ img: RasterImage) -> [PixelLabel] {
        let w = img.width, h = img.height
        let rgba = img.rgba

        var luma = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let r = rgba[i * 4 + 0]
            let g = rgba[i * 4 + 1]
            let b = rgba[i * 4 + 2]
            luma[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        var out = [PixelLabel](repeating: PixelLabel(classID: 0, confidence: 1),
                               count: w * h)

        for y in 0..<h {
            for x in 0..<w {
                let idx = y * w + x
                var gx: Float = 0
                var gy: Float = 0
                if x > 0 && x < w - 1 && y > 0 && y < h - 1 {
                    let tl = luma[(y - 1) * w + (x - 1)]
                    let tc = luma[(y - 1) * w +  x     ]
                    let tr = luma[(y - 1) * w + (x + 1)]
                    let ml = luma[ y      * w + (x - 1)]
                    let mr = luma[ y      * w + (x + 1)]
                    let bl = luma[(y + 1) * w + (x - 1)]
                    let bc = luma[(y + 1) * w +  x     ]
                    let br = luma[(y + 1) * w + (x + 1)]
                    gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)
                    gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr)
                }
                let g = (gx * gx + gy * gy).squareRoot()

                let classID: UInt8
                let conf: Float
                if g >= edgeThreshold {
                    classID = 1
                    conf = min(1.0, 0.5 + (g - edgeThreshold) / edgeThreshold)
                } else if g >= textureThreshold {
                    classID = 2
                    conf = 0.5 + (g - textureThreshold)
                         / (edgeThreshold - textureThreshold) * 0.4
                } else {
                    classID = 0
                    conf = min(1.0, 0.6 + (textureThreshold - g)
                         / textureThreshold * 0.4)
                }
                out[idx] = PixelLabel(classID: classID, confidence: conf)
            }
        }
        return out
    }
}
