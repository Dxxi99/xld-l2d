import Foundation
import AVFoundation
import Vision
import QuartzCore

final class TrackerState: @unchecked Sendable {
    var emaYaw: Double = 0
    var emaPitch: Double = 0
    var emaRoll: Double = 0
    var lastFrameTime: CFTimeInterval = 0
    var smoothedFPS: Double = 0
}

@MainActor
final class CameraTracker: NSObject, ObservableObject {
    @Published var isRunning: Bool = false
    @Published var errorMessage: String?

    var onAngles: ((Double, Double, Double) -> Void)?
    var onStatus: ((Bool, Double) -> Void)?

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "xld.camera", qos: .userInitiated)
    private var device: AVCaptureDevice?
    private var configuredFPS: Int = 60
    private var isStarting: Bool = false

    // 상태 (nonisolated 콜백에서 접근)
    nonisolated let state = TrackerState()
    private let emaAlpha: Double = 0.35

    func start(targetFPS: Int = 60) {
        print("[CameraTracker] start() called, targetFPS=\(targetFPS)")
        guard !isRunning && !isStarting else {
            print("[CameraTracker] already running/starting, skip")
            return
        }
        isStarting = true
        configuredFPS = targetFPS
        errorMessage = nil

        let status = AVCaptureDevice.authorizationStatus(for: .video)
        print("[CameraTracker] auth status=\(status.rawValue) (0=notDetermined,1=restricted,2=denied,3=authorized)")
        switch status {
        case .authorized:
            beginSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    if granted {
                        self?.beginSession()
                    } else {
                        self?.errorMessage = "Camera access denied"
                        self?.isStarting = false
                    }
                }
            }
        default:
            errorMessage = "Camera access denied (System Settings)"
            isStarting = false
        }
    }

    func stop() {
        guard isRunning else { isStarting = false; return }
        let s = session
        queue.async { if s.isRunning { s.stopRunning() } }
        isRunning = false
        isStarting = false
    }

    private func beginSession() {
        let fps = configuredFPS
        queue.async { [weak self] in
            guard let self else { return }
            print("[CameraTracker] setupSession on bg queue...")
            self.setupSession(targetFPS: fps)
            print("[CameraTracker] startRunning...")
            if !self.session.isRunning { self.session.startRunning() }
            print("[CameraTracker] startRunning done, isRunning=\(self.session.isRunning)")
            Task { @MainActor in
                self.isRunning = true
                self.isStarting = false
                print("[CameraTracker] fully started")
            }
        }
    }

    private func setupSession(targetFPS: Int) {
        print("[CameraTracker] setupSession begin")
        session.beginConfiguration()
        defer {
            session.commitConfiguration()
            print("[CameraTracker] setupSession committed, inputs=\(session.inputs.count), outputs=\(session.outputs.count)")
        }

        session.sessionPreset = .hd1280x720

        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
              ?? AVCaptureDevice.default(for: .video) else {
            errorMessage = "No camera device"
            print("[CameraTracker] ERROR: no camera device")
            return
        }
        device = dev
        print("[CameraTracker] device=\(dev.localizedName)")

        print("[CameraTracker] creating input...")
        do {
            let inp = try AVCaptureDeviceInput(device: dev)
            if session.canAddInput(inp) {
                session.addInput(inp)
                print("[CameraTracker] input added")
            }
        } catch {
            errorMessage = "Input error: \(error.localizedDescription)"
            print("[CameraTracker] input error: \(error)")
            return
        }

        let out = AVCaptureVideoDataOutput()
        out.alwaysDiscardsLateVideoFrames = true
        out.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        print("[CameraTracker] creating output...")
        out.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(out) {
            session.addOutput(out)
            print("[CameraTracker] output added")
        }

        // 지원 FPS 범위 확인 후 클램프
        let ranges = dev.activeFormat.videoSupportedFrameRateRanges
        let supportedMax = ranges.map { $0.maxFrameRate }.max() ?? 30
        let supportedMin = ranges.map { $0.minFrameRate }.min() ?? 15
        let actualFPS = max(supportedMin, min(Double(targetFPS), supportedMax))
        print("[CameraTracker] supported fps: \(supportedMin)~ \(supportedMax), requested=\(targetFPS), using=\(actualFPS)")

        do {
            try dev.lockForConfiguration()
            let dur = CMTime(value: 1, timescale: CMTimeScale(actualFPS))
            dev.activeVideoMinFrameDuration = dur
            dev.activeVideoMaxFrameDuration = dur
            dev.unlockForConfiguration()
            print("[CameraTracker] fps set to \(actualFPS)")
        } catch {
            print("[CameraTracker] fps config error: \(error)")
        }
    }
}

extension CameraTracker: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let req = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pb,
                                            orientation: .upMirrored,
                                            options: [:])
        do {
            try handler.perform([req])
        } catch {
            return
        }

        let st = self.state
        let now = CACurrentMediaTime()
        var fps: Double = 0
        if st.lastFrameTime > 0 {
            let dt = now - st.lastFrameTime
            let inst = 1.0 / max(dt, 1e-6)
            st.smoothedFPS = st.smoothedFPS == 0 ? inst : st.smoothedFPS * 0.9 + inst * 0.1
            fps = st.smoothedFPS
        }
        st.lastFrameTime = now

        guard let results = req.results, let face = results.first else {
            Task { @MainActor in self.onStatus?(false, fps) }
            return
        }

        let yawRad = face.yaw?.doubleValue ?? 0
        let pitchRad = face.pitch?.doubleValue ?? 0
        let rollRad = face.roll?.doubleValue ?? 0

        let yawDeg = yawRad * 180.0 / .pi
        let pitchDeg = pitchRad * 180.0 / .pi
        let rollDeg = rollRad * 180.0 / .pi

        let alpha = self.emaAlpha
        st.emaYaw   += alpha * (yawDeg   - st.emaYaw)
        st.emaPitch += alpha * (pitchDeg - st.emaPitch)
        st.emaRoll  += alpha * (rollDeg  - st.emaRoll)

        let fy = st.emaYaw
        let fp = st.emaPitch
        let fr = st.emaRoll

        Task { @MainActor in
            self.onAngles?(fy, fp, fr)
            self.onStatus?(true, fps)
        }
    }
}
