import Foundation
import CoreGraphics
import ImageIO

public struct RasterImage: Sendable {
    public let width: Int
    public let height: Int
    public let rgba: [Float]

    public init(width: Int, height: Int, rgba: [Float]) {
        precondition(rgba.count == width * height * 4)
        self.width = width
        self.height = height
        self.rgba = rgba
    }
}

public enum ImageIOError: Error { case loadFailed, saveFailed }

public enum ImageIO {
    public static func load(url: URL) throws -> RasterImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg  = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw ImageIOError.loadFailed
        }
        let w = cg.width, h = cg.height
        let bytesPerRow = w * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * h)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ok: Bool = pixels.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress,
                                      width: w, height: h,
                                      bitsPerComponent: 8,
                                      bytesPerRow: bytesPerRow,
                                      space: cs,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw ImageIOError.loadFailed }

        var out = [Float](repeating: 0, count: w * h * 4)
        for i in 0..<out.count { out[i] = Float(pixels[i]) / 255.0 }
        return RasterImage(width: w, height: h, rgba: out)
    }

    public static func save(image: RasterImage, url: URL) throws {
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<bytes.count {
            let v = image.rgba[i] * 255.0
            bytes[i] = UInt8(max(0, min(255, v)))
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let cg: CGImage? = bytes.withUnsafeMutableBytes { raw -> CGImage? in
            guard let ctx = CGContext(data: raw.baseAddress,
                                      width: w, height: h,
                                      bitsPerComponent: 8,
                                      bytesPerRow: w * 4,
                                      space: cs,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            return ctx.makeImage()
        }
        guard let img = cg,
              let dest = CGImageDestinationCreateWithURL(url as CFURL,
                                                          "public.png" as CFString, 1, nil)
        else { throw ImageIOError.saveFailed }
        CGImageDestinationAddImage(dest, img, nil)
        if !CGImageDestinationFinalize(dest) { throw ImageIOError.saveFailed }
    }
}
