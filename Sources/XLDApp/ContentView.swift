import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState

    // panel sizes (draggable via SplitView)
    @State private var layersWidth: CGFloat = 220
    @State private var inspectorWidth: CGFloat = 280
    @State private var timelineHeight: CGFloat = 180

    var body: some View {
        VStack(spacing: 0) {
            // ─── Toolbar ───
            Toolbar(appState: appState)
                .frame(height: 38)
                .background(Color(white: 0.13))
            Divider().background(Color(white: 0.25))

            // ─── Main 3-way split ───
            HSplitView {
                LayersPanel()
                    .frame(minWidth: 160, idealWidth: layersWidth, maxWidth: 400)
                    .background(Color(white: 0.11))

                CanvasPanel()
                    .frame(minWidth: 500)
                    .background(Color(white: 0.06))

                InspectorPanel()
                    .frame(minWidth: 220, idealWidth: inspectorWidth, maxWidth: 420)
                    .background(Color(white: 0.11))
            }

            // ─── Status bar ───
            StatusBar(appState: appState)
                .frame(height: 24)
                .background(Color(white: 0.13))
        }
        .background(Color(white: 0.10))
        .overlay(alignment: .bottomTrailing) {
            if appState.showTrackingDebug {
                TrackingDebugView(store: appState.debugStore)
                    .frame(width: 520, height: 440)
                    .background(Color(white: 0.08))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.orange, lineWidth: 2))
                    .cornerRadius(8)
                    .padding(16)
            }
        }
    }
}

// ─── Toolbar (top) ───
struct Toolbar: View {
    @ObservedObject var appState: AppState

    var body: some View {
        HStack(spacing: 16) {
            Button {
                openImagePanel()
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "folder")
                    Text("Open Image…")
                        .font(.system(size: 12))
                }
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(RoundedRectangle(cornerRadius: 4).fill(Color(white: 0.20)))
            .disabled(appState.isProcessing)

            Divider().frame(height: 16)

            HStack(spacing: 6) {
                ToolbarIcon(systemName: "cursorarrow", label: "Select")
                ToolbarIcon(systemName: "hand.draw", label: "Pan")
                ToolbarIcon(systemName: "point.topleft.down.to.point.bottomright.curvepath", label: "Mesh")
            }

            Spacer()

            Text("XLD-L2D")
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(.white.opacity(0.85))

            Spacer()

            HStack(spacing: 6) {
                ToolbarIcon(systemName: "record.circle", label: "Record")
                ToolbarIcon(systemName: "play.fill", label: "Play")
            }
        }
        .padding(.horizontal, 10)
    }

    private func openImagePanel() {
        print("[Toolbar] openImagePanel() called")
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .tiff, .heic]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.message = "Select an illustration (PNG)"
        panel.prompt = "Open"
        let response = panel.runModal()
        print("[Toolbar] panel response=\(response.rawValue)")
        if response == .OK, let url = panel.url {
            print("[Toolbar] selected: \(url.path)")
            appState.openImage(url)
        }
    }
}

struct ToolbarIcon: View {
    let systemName: String
    let label: String
    var body: some View {
        Image(systemName: systemName)
            .font(.system(size: 13))
            .frame(width: 26, height: 22)
            .background(RoundedRectangle(cornerRadius: 4).fill(Color(white: 0.18)))
            .foregroundColor(.white.opacity(0.8))
            .help(label)
    }
}

// ─── Status bar (bottom) ───
struct StatusBar: View {
    @ObservedObject var appState: AppState
    var body: some View {
        HStack(spacing: 12) {
            Text(appState.statusMessage)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.white.opacity(0.7))
                .lineLimit(1)

            if appState.isProcessing {
                ProgressView(value: Double(appState.progressStep),
                             total: Double(appState.progressTotal))
                    .frame(width: 140)
                    .tint(.cyan)
            }

            Spacer()

            Text(String(format: "%.1f fps", appState.fps))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.green.opacity(0.8))
        }
        .padding(.horizontal, 10)
    }
}
