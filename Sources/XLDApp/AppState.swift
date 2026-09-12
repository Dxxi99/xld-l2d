import Foundation
import SwiftUI
import XLDKit

@MainActor
final class AppState: ObservableObject {
    // Model
    @Published var model: XLDRuntimeModel?
    @Published var sourceImage: NSImage?

    // Pose
    @Published var yawDeg: Double = 0
    @Published var pitchDeg: Double = 0

    // Status
    @Published var statusMessage: String = "Ready"
    @Published var fps: Double = 0
    @Published var isProcessing: Bool = false
    @Published var progressStep: Int = 0
    @Published var progressTotal: Int = 7
    @Published var lastError: String?

    // Panels
    @Published var selectedLayer: String? = nil
    @Published var inspectorTab: InspectorTab = .properties

    let projectsDir: URL = {
        let d = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("XLD-L2D/projects")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()

    enum InspectorTab: String, CaseIterable, Identifiable {
        case properties = "Properties"
        case behaviors  = "Behaviors"
        case filters    = "Filters"
        case object     = "Object"
        var id: String { rawValue }
    }

    func resetPose() { yawDeg = 0; pitchDeg = 0 }

    func openImage(_ url: URL) {
        guard !isProcessing else { return }
        isProcessing = true
        lastError = nil
        progressStep = 0
        progressTotal = 7
        statusMessage = "Processing..."

        let projectName = url.deletingPathExtension().lastPathComponent
        let modelDir = projectsDir.appendingPathComponent(projectName)

        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                try ModelLoader.runPipeline(inputImage: url, outModelDir: modelDir) { step, total, label in
                    Task { @MainActor in
                        self?.progressStep = step
                        self?.progressTotal = total
                        self?.statusMessage = "[\(step)/\(total)] \(label)"
                    }
                }
                let m = try XLDRuntimeModel.load(from: modelDir)
                let img = NSImage(contentsOf: modelDir.appendingPathComponent("source.png"))
                await MainActor.run {
                    self?.model = m
                    self?.sourceImage = img
                    self?.isProcessing = false
                    self?.statusMessage = "Loaded \(m.mesh.vertexCount)v \(m.mesh.triangleCount)t"
                }
            } catch {
                await MainActor.run {
                    self?.isProcessing = false
                    self?.lastError = "\(error)"
                    self?.statusMessage = "Error: \(error.localizedDescription)"
                }
            }
        }
    }
}
