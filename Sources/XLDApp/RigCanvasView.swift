import SwiftUI
import simd
import XLDKit
import CoreGraphics

/// Renders runtime model with yaw/pitch via Core Graphics rasterization.
/// Fast enough for 464k triangles because CG is C-optimized.
public struct RigCanvasView: View {
    public let model: XLDRuntimeModel
    public let triangleColors: [CGColor]
    public let yawDeg: Double
    public let pitchDeg: Double

    public init(model: XLDRuntimeModel,
                triangleColors: [CGColor],
                yawDeg: Double,
                pitchDeg: Double) {
        self.model = model
        self.triangleColors = triangleColors
        self.yawDeg = yawDeg
        self.pitchDeg = pitchDeg
    }

    public var body: some View {
        GeometryReader { geo in
            Image(nsImage: rasterize(size: geo.size))
                .resizable()
                .interpolation(.none)
                .aspectRatio(contentMode: .fit)
        }
        .background(Color.black)
    }

    private func rasterize(size: CGSize) -> NSImage {
        let W = CGFloat(model.width)
        let H = CGFloat(model.height)
        let scale = min(size.width / W, size.height / H)
        let ox = (size.width  - W * scale) / 2
        let oy = (size.height - H * scale) / 2

        let pw = Int(size.width)
        let ph = Int(size.height)
        guard pw > 0, ph > 0 else {
            return NSImage(size: NSSize(width: 1, height: 1))
        }

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil,
                                  width: pw, height: ph,
                                  bitsPerComponent: 8,
                                  bytesPerRow: pw * 4,
                                  space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            return NSImage(size: NSSize(width: 1, height: 1))
        }

        ctx.setShouldAntialias(false)   // kill seam AA
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: pw, height: ph))

        // offsets
        let offs = model.rig.offsets(yawDeg: Float(yawDeg), pitchDeg: Float(pitchDeg))
        let verts = model.mesh.vertices
        let tris = model.mesh.triangles

        // build path: batch by color? For now, per-tri path
        // Use CGContext path and fill per tri (still fast in CG)
        for i in 0..<tris.count {
            let tri = tris[i]
            let i0 = Int(tri.x), i1 = Int(tri.y), i2 = Int(tri.z)
            let p0 = verts[i0] + offs[i0]
            let p1 = verts[i1] + offs[i1]
            let p2 = verts[i2] + offs[i2]

            // expand 1.15x around centroid (fill tiny gaps)
            let cx = (p0.x + p1.x + p2.x) / 3
            let cy = (p0.y + p1.y + p2.y) / 3
            let k: Float = 1.08
            let q0 = SIMD2<Float>(cx + (p0.x - cx) * k, cy + (p0.y - cy) * k)
            let q1 = SIMD2<Float>(cx + (p1.x - cx) * k, cy + (p1.y - cy) * k)
            let q2 = SIMD2<Float>(cx + (p2.x - cx) * k, cy + (p2.y - cy) * k)

            // CG Y-flip: mesh y (0=top) -> CG y (0=bottom)
            let x0 = CGFloat(q0.x) * scale + ox
            let y0 = size.height - (CGFloat(q0.y) * scale + oy)
            let x1 = CGFloat(q1.x) * scale + ox
            let y1 = size.height - (CGFloat(q1.y) * scale + oy)
            let x2 = CGFloat(q2.x) * scale + ox
            let y2 = size.height - (CGFloat(q2.y) * scale + oy)

            ctx.beginPath()
            ctx.move(to: CGPoint(x: x0, y: y0))
            ctx.addLine(to: CGPoint(x: x1, y: y1))
            ctx.addLine(to: CGPoint(x: x2, y: y2))
            ctx.closePath()
            ctx.setFillColor(triangleColors[i])
            ctx.fillPath()
        }

        guard let cg = ctx.makeImage() else {
            return NSImage(size: NSSize(width: 1, height: 1))
        }
        return NSImage(cgImage: cg, size: NSSize(width: pw, height: ph))
    }
}
