import Foundation
import Network
import simd
import XLDKit

// MediaPipe blendshape 결과 (52개 중 우리가 쓰는 것만)
struct FaceBlendshapes: Sendable {
    var eyeBlinkL: Float = 0
    var eyeBlinkR: Float = 0
    var jawOpen: Float = 0
    var mouthSmileL: Float = 0
    var mouthSmileR: Float = 0
    var mouthFrownL: Float = 0
    var mouthFrownR: Float = 0
    var browDownL: Float = 0
    var browDownR: Float = 0
    var browOuterUpL: Float = 0
    var browOuterUpR: Float = 0
    var eyeLookInL: Float = 0
    var eyeLookOutL: Float = 0
    var eyeLookUpL: Float = 0
    var eyeLookDownL: Float = 0
    var eyeLookInR: Float = 0
    var eyeLookOutR: Float = 0
    var eyeLookUpR: Float = 0
    var eyeLookDownR: Float = 0
    var mouthPucker: Float = 0
    var mouthFunnel: Float = 0

    var yawRad: Float = 0
    var pitchRad: Float = 0
    var rollRad: Float = 0

    var hasFace: Bool = false
    var timestamp: Double = 0
}

final class MediaPipeEMAState: @unchecked Sendable {
    var eyeBlinkL: Float = 0
    var eyeBlinkR: Float = 0
    var jawOpen: Float = 0
    var yaw: Float = 0
    var pitch: Float = 0
    var initialized: Bool = false
}

@MainActor
final class MediaPipeClient: ObservableObject {
    @Published var isRunning: Bool = false
    @Published var errorMessage: String?
    @Published var lastBlendshape: FaceBlendshapes = FaceBlendshapes()

    var onBlendshapes: ((FaceBlendshapes) -> Void)?

    private let queue = DispatchQueue(label: "xld.mediapipe", qos: .userInitiated)
    private var listener: NWListener?
    private var daemonProcess: Process?
    private let port: UInt16 = 5005
    private let ema = MediaPipeEMAState()
    private let emaAlpha: Float = 0.35  // 클수록 빠름, 작을수록 부드러움

    // 프로젝트 루트 (daemon 실행용)
    private let projectRoot: URL

    init() {
        // 현재 실행 파일 기준 프로젝트 루트 추정
        // swift run 은 .build/debug/XLDApp 이므로 3단계 위
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        self.projectRoot = cwd
    }

    func start() {
        print("[MediaPipe] start, root=\(projectRoot.path)")
        guard !isRunning else { return }

        // 1) UDP listener
        do {
            let params = NWParameters.udp
            params.allowLocalEndpointReuse = true
            let l = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
            l.newConnectionHandler = { [weak self] conn in
                conn.start(queue: self?.queue ?? .main)
                self?.receive(on: conn)
            }
            l.stateUpdateHandler = { state in
                print("[MediaPipe] listener state: \(state)")
            }
            l.start(queue: queue)
            self.listener = l
        } catch {
            errorMessage = "listener: \(error.localizedDescription)"
            print("[MediaPipe] listener error: \(error)")
            return
        }

        // 2) Python daemon 시작
        startDaemon()

        isRunning = true
    }

    func stop() {
        listener?.cancel()
        listener = nil
        if let p = daemonProcess, p.isRunning {
            p.terminate()
        }
        daemonProcess = nil
        isRunning = false
    }

    private func startDaemon() {
        let venvPython = projectRoot.appendingPathComponent(".venv/bin/python3").path
        let script = projectRoot.appendingPathComponent("tools/mediapipe_daemon.py").path

        guard FileManager.default.fileExists(atPath: venvPython) else {
            errorMessage = "venv python not found: \(venvPython)"
            print("[MediaPipe] \(errorMessage!)")
            return
        }
        guard FileManager.default.fileExists(atPath: script) else {
            errorMessage = "daemon script not found: \(script)"
            print("[MediaPipe] \(errorMessage!)")
            return
        }

        let task = Process()
        task.executableURL = URL(fileURLWithPath: venvPython)
        task.arguments = [script, "--port", "\(port)"]
        task.currentDirectoryURL = projectRoot

        // 로그 파이프 (stdout/stderr)
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { fh in
            let data = fh.availableData
            if let s = String(data: data, encoding: .utf8), !s.isEmpty {
                print("[daemon] \(s.trimmingCharacters(in: .whitespacesAndNewlines))")
            }
        }

        do {
            try task.run()
            daemonProcess = task
            print("[MediaPipe] daemon pid=\(task.processIdentifier)")
        } catch {
            errorMessage = "daemon launch: \(error.localizedDescription)"
            print("[MediaPipe] \(errorMessage!)")
        }
    }

    nonisolated private func receive(on conn: NWConnection) {
        conn.receiveMessage { [weak self] data, _, _, error in
            if let error = error {
                print("[MediaPipe] recv error: \(error)")
                return
            }
            if let data = data, !data.isEmpty {
                self?.handlePacket(data)
            }
            self?.receive(on: conn)
        }
    }

    private nonisolated func handlePacket(_ data: Data) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        guard let faces = json["faces"] as? Int, faces > 0 else { return }
        guard let bs = json["blendshapes"] as? [String: Double] else { return }

        var out = FaceBlendshapes()
        out.hasFace = true
        out.timestamp = json["t"] as? Double ?? 0

        func get(_ k: String) -> Float { Float(bs[k] ?? 0) }

        out.eyeBlinkL = get("eyeBlinkLeft")
        out.eyeBlinkR = get("eyeBlinkRight")
        out.jawOpen = get("jawOpen")
        out.mouthSmileL = get("mouthSmileLeft")
        out.mouthSmileR = get("mouthSmileRight")
        out.mouthFrownL = get("mouthFrownLeft")
        out.mouthFrownR = get("mouthFrownRight")
        out.browDownL = get("browDownLeft")
        out.browDownR = get("browDownRight")
        out.browOuterUpL = get("browOuterUpLeft")
        out.browOuterUpR = get("browOuterUpRight")
        out.eyeLookInL = get("eyeLookInLeft")
        out.eyeLookOutL = get("eyeLookOutLeft")
        out.eyeLookUpL = get("eyeLookUpLeft")
        out.eyeLookDownL = get("eyeLookDownLeft")
        out.eyeLookInR = get("eyeLookInRight")
        out.eyeLookOutR = get("eyeLookOutRight")
        out.eyeLookUpR = get("eyeLookUpRight")
        out.eyeLookDownR = get("eyeLookDownRight")
        out.mouthPucker = get("mouthPucker")
        out.mouthFunnel = get("mouthFunnel")

        // head transform matrix → yaw/pitch/roll
        // MediaPipe numpy 반환은 row-major (4x4)
        if let m = json["matrix"] as? [Double], m.count >= 16 {
            // row-major: m[0..3]=row0, m[4..7]=row1, m[8..11]=row2, m[12..15]=row3
            let r00 = Float(m[0]), r01 = Float(m[1]), r02 = Float(m[2])
            let r10 = Float(m[4]), r11 = Float(m[5]), r12 = Float(m[6])
            let r20 = Float(m[8]), r21 = Float(m[9]), r22 = Float(m[10])

            // ZYX Euler (head pose 표준)
            let sy = sqrt(r00*r00 + r10*r10)
            if sy > 1e-6 {
                // yaw = Y축, pitch = X축, roll = Z축
                out.yawRad   = atan2(-r20, sy)
                out.pitchRad = atan2(r21, r22)
                out.rollRad  = atan2(r10, r00)
            }

        }

        // EMA 스무딩 (nonisolated context)
        let a = self.emaAlpha
        if !self.ema.initialized {
            self.ema.eyeBlinkL = out.eyeBlinkL
            self.ema.eyeBlinkR = out.eyeBlinkR
            self.ema.jawOpen = out.jawOpen
            self.ema.yaw = out.yawRad
            self.ema.pitch = out.pitchRad
            self.ema.initialized = true
        } else {
            self.ema.eyeBlinkL += a * (out.eyeBlinkL - self.ema.eyeBlinkL)
            self.ema.eyeBlinkR += a * (out.eyeBlinkR - self.ema.eyeBlinkR)
            self.ema.jawOpen    += a * (out.jawOpen    - self.ema.jawOpen)
            self.ema.yaw        += a * (out.yawRad     - self.ema.yaw)
            self.ema.pitch      += a * (out.pitchRad   - self.ema.pitch)
        }

        var smoothed = out
        smoothed.eyeBlinkL = self.ema.eyeBlinkL
        smoothed.eyeBlinkR = self.ema.eyeBlinkR
        smoothed.jawOpen = self.ema.jawOpen
        smoothed.yawRad = self.ema.yaw
        smoothed.pitchRad = self.ema.pitch

        Task { @MainActor in
            self.lastBlendshape = smoothed
            self.onBlendshapes?(smoothed)
        }
    }
}
