import SwiftUI
import simd
import XLDKit

public struct ContentView: View {
    @StateObject private var state = RigState()

    @State private var model: XLDRuntimeModel?
    @State private var triangleColors: [CGColor] = []
    @State private var loadError: String?

    public init() {}

    public var body: some View {
        HStack(spacing: 0) {
            ZStack {
                if let m = model {
                    MetalRigView(model: m,
                                 yawDeg: state.yawDeg,
                                 pitchDeg: state.pitchDeg,
                                 sourceImageURL: m.sourceImageURL,
                                 onFPS: { f in
                                     DispatchQueue.main.async {
                                         state.fps = f
                                     }
                                 })
                } else if let err = loadError {
                    VStack {
                        Text("Load failed").font(.title)
                        Text(err).font(.caption).foregroundColor(.red)
                    }
                } else {
                    ProgressView("Loading model...")
                }
                VStack {
                    Spacer()
                    HStack {
                        Text(String(format: "%.1f fps", state.fps))
                            .font(.system(.caption, design: .monospaced))
                            .padding(6)
                            .background(Color.black.opacity(0.6))
                            .foregroundColor(.green)
                        Text(state.statusText)
                            .font(.system(.caption2, design: .monospaced))
                            .padding(6)
                            .background(Color.black.opacity(0.6))
                            .foregroundColor(.cyan)
                        Spacer()
                    }
                }
                .padding(8)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            VStack(alignment: .leading, spacing: 20) {
                Text("XLD-L2D").font(.title).bold()

                Divider()

                VStack(alignment: .leading) {
                    Text("Yaw: \(String(format: "%+.1f", state.yawDeg))°")
                    Slider(value: $state.yawDeg, in: -30...30)
                }

                VStack(alignment: .leading) {
                    Text("Pitch: \(String(format: "%+.1f", state.pitchDeg))°")
                    Slider(value: $state.pitchDeg, in: -15...15)
                }

                Button("Reset") {
                    state.yawDeg = 0
                    state.pitchDeg = 0
                }

                Spacer()

                Text("triangles: \(model?.mesh.triangleCount ?? 0)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            .padding(24)
            .frame(width: 320)
            .background(Color(NSColor.windowBackgroundColor))
        }
        .onAppear { load() }
    }

    private func load() {
        do {
            let dir: URL
            if let envPath = ProcessInfo.processInfo.environment["XLD_MODEL_DIR"] {
                dir = URL(fileURLWithPath: envPath)
                print("[ContentView] using XLD_MODEL_DIR=\(envPath)")
            } else {
                dir = FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("XLD-L2D/assets/real/swift_cache/model_v1")
            }
            let m = try XLDRuntimeModel.load(from: dir)
            self.model = m
            self.state.statusText = "loaded \(m.mesh.vertexCount)v \(m.mesh.triangleCount)t"

            let imgURL = dir.appendingPathComponent("source.png")
            if let ns = NSImage(contentsOf: imgURL) {
                self.triangleColors = Self.sampleColors(image: ns, model: m)
            } else {
                self.triangleColors = [CGColor](repeating: CGColor(gray: 0.5, alpha: 1.0),
                                                count: m.mesh.triangleCount)
            }
        } catch {
            self.loadError = "\(error)"
        }
    }

    private static func sampleColors(image: NSImage, model: XLDRuntimeModel) -> [CGColor] {
        guard let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff) else {
            return [CGColor](repeating: CGColor(gray: 0.5, alpha: 1.0),
                             count: model.mesh.triangleCount)
        }
        let W = bitmap.pixelsWide
        let H = bitmap.pixelsHigh
        let verts = model.mesh.vertices
        let tris = model.mesh.triangles
        var colors: [CGColor] = []
        colors.reserveCapacity(tris.count)
        for tri in tris {
            let i0 = Int(tri.x), i1 = Int(tri.y), i2 = Int(tri.z)
            let cx = (verts[i0].x + verts[i1].x + verts[i2].x) / 3
            let cy = (verts[i0].y + verts[i1].y + verts[i2].y) / 3
            let px = max(0, min(W-1, Int(cx)))
            let py = max(0, min(H-1, Int(cy)))
            if let c = bitmap.colorAt(x: px, y: py) {
                colors.append(CGColor(red: c.redComponent, green: c.greenComponent,
                                      blue: c.blueComponent, alpha: c.alphaComponent))
            } else {
                colors.append(CGColor(gray: 0.5, alpha: 1.0))
            }
        }
        return colors
    }
}
