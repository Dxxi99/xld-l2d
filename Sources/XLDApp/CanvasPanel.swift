import SwiftUI
import UniformTypeIdentifiers
import XLDKit

struct CanvasPanel: View {
    @EnvironmentObject var appState: AppState
    @State private var isDropTargeted = false

    var body: some View {
        ZStack {
            Color(white: 0.05)

            if let m = appState.model {
                MetalRigView(model: m,
                             yawDeg: appState.yawDeg,
                             pitchDeg: appState.pitchDeg,
                             eyeOpen_L: appState.eyeOpen_L,
                             eyeOpen_R: appState.eyeOpen_R,
                             mouthOpen: appState.mouthOpen,
                             eyeL_X: appState.eyeL_regionX,
                             eyeL_Y: appState.eyeL_regionY,
                             eyeR_X: appState.eyeR_regionX,
                             eyeR_Y: appState.eyeR_regionY,
                             mouth_X: appState.mouth_regionX,
                             mouth_Y: appState.mouth_regionY,
                             eyeR_N: appState.eye_radiusN,
                             mouthR_N: appState.mouth_radiusN,
                             sourceImageURL: m.sourceImageURL,
                             onFPS: { f in
                                 DispatchQueue.main.async { appState.fps = f }
                             })
            } else {
                EmptyCanvas()
            }

            // drop overlay
            if isDropTargeted {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8, 6]))
                    .padding(20)
                    .overlay(
                        Text("Drop image to process")
                            .font(.system(size: 14, weight: .semibold))
                            .foregroundColor(.accentColor)
                    )
            }
        }
        .onDrop(of: [.fileURL], isTargeted: $isDropTargeted) { providers in
            handleDrop(providers)
        }
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url else { return }
            let ext = url.pathExtension.lowercased()
            guard ["png", "jpg", "jpeg", "tiff", "heic"].contains(ext) else { return }
            Task { @MainActor in
                appState.openImage(url)
            }
        }
        return true
    }
}

struct EmptyCanvas: View {
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 56, weight: .ultraLight))
                .foregroundColor(.white.opacity(0.25))
            Text("Open an image to begin")
                .font(.system(size: 15))
                .foregroundColor(.white.opacity(0.5))
            Text("or drop a PNG here")
                .font(.system(size: 12))
                .foregroundColor(.white.opacity(0.35))
        }
    }
}
