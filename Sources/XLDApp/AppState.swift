import Foundation
import QuartzCore
import SwiftUI
import XLDKit

@MainActor
final class AppState: ObservableObject {
    @Published var showTrackingDebug: Bool = false
    let debugStore = TrackingDebugStore()
    let mediaPipeClient = MediaPipeClient()

    // Model
    @Published var model: XLDRuntimeModel?
    @Published var sourceImage: NSImage?

    // Pose
    @Published var yawDeg: Double = 0
    @Published var pitchDeg: Double = 0

    // Tracking
    @Published var trackingEnabled: Bool = false
    @Published var trackingFPS: Int = 60
    @Published var faceDetected: Bool = false
    @Published var cameraFPS: Double = 0
    let cameraTracker = CameraTracker()

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

    private var lastRawLog: CFTimeInterval = 0

    // Face feature (트래킹 → 캐릭터)
    @Published var eyeOpen_L: Double = 1.0
    @Published var eyeOpen_R: Double = 1.0
    @Published var mouthOpen: Double = 0.0

    // 얼굴 부위 위치 (정규화 0~1, 캐릭터 이미지 기준)
    @Published var eyeL_regionX: Double = 0.38
    @Published var eyeL_regionY: Double = 0.22
    @Published var eyeR_regionX: Double = 0.62
    @Published var eyeR_regionY: Double = 0.22
    @Published var mouth_regionX: Double = 0.50
    @Published var mouth_regionY: Double = 0.30
    @Published var eye_radiusN: Double = 0.06
    @Published var mouth_radiusN: Double = 0.08

    func resetPose() { yawDeg = 0; pitchDeg = 0 }

    @Published var mediaPipeEnabled: Bool = false

    func toggleMediaPipe() {
        if mediaPipeEnabled {
            mediaPipeClient.stop()
            mediaPipeEnabled = false
        } else {
            mediaPipeClient.onBlendshapes = { [weak self] bs in
                guard let self else { return }
                // 눈 blink remap: MediaPipe 0.1~0.75 → 0~1
                func remapBlink(_ v: Float) -> Float {
                    let a = max(0, v * 1.4)
                    return min(1, a)
                }
                let blinkL = remapBlink(bs.eyeBlinkL)
                let blinkR = remapBlink(bs.eyeBlinkR)
                self.eyeOpen_L = Double(1.0 - blinkL)
                self.eyeOpen_R = Double(1.0 - blinkR)
                if Int(Date().timeIntervalSince1970 * 3) % 3 == 0 {
                    print(String(format: "[app-open] L=%.2f R=%.2f mouth=%.2f",
                                 self.eyeOpen_L, self.eyeOpen_R, self.mouthOpen))
                }

                // 입: jawOpen 0~0.6 → 0~1
                let jaw = min(1.0, bs.jawOpen / 0.6)
                self.mouthOpen = Double(jaw)
                // head 각도 (deg)
                let yawDeg = Double(bs.yawRad) * 180.0 / .pi
                let pitchDeg = Double(bs.pitchRad) * 180.0 / .pi
                // MediaPipe matrix yaw/pitch 부호·스케일 조정
                let yawAmp: Double = 1.4
                let pitchAmp: Double = 1.4
                let y = yawDeg * yawAmp
                let p = pitchDeg * pitchAmp
                self.yawDeg = max(-30, min(30, y))
                self.pitchDeg = max(-30, min(30, p))
            }
            mediaPipeClient.start()
            mediaPipeEnabled = true
        }
    }

    // Vision calibration 폐기됨 (MediaPipe 는 calibration 불필요)

    func toggleTracking() {
        if trackingEnabled {
            cameraTracker.stop()
            trackingEnabled = false
            faceDetected = false
            cameraFPS = 0
        } else {




            cameraTracker.start(targetFPS: trackingFPS)
            trackingEnabled = true
        }
    }

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
                    // anime-face-detector 자동 region 반영
                    if m.meta.hasRegionAuto {
                        self?.eyeL_regionX = Double(m.meta.regionEyeLX)
                        self?.eyeL_regionY = Double(m.meta.regionEyeLY)
                        self?.eyeR_regionX = Double(m.meta.regionEyeRX)
                        self?.eyeR_regionY = Double(m.meta.regionEyeRY)
                        self?.mouth_regionX = Double(m.meta.regionMouthX)
                        self?.mouth_regionY = Double(m.meta.regionMouthY)
                        self?.eye_radiusN = Double(m.meta.regionEyeRadius)
                        self?.mouth_radiusN = Double(m.meta.regionMouthRadius)
                        print(String(format: "[face-region] auto: eyeL(%.3f,%.3f) eyeR(%.3f,%.3f) mouth(%.3f,%.3f) r=%.3f/%.3f",
                                     m.meta.regionEyeLX, m.meta.regionEyeLY,
                                     m.meta.regionEyeRX, m.meta.regionEyeRY,
                                     m.meta.regionMouthX, m.meta.regionMouthY,
                                     m.meta.regionEyeRadius, m.meta.regionMouthRadius))
                    }
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
