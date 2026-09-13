import Foundation
import AVFoundation
import Vision
import QuartzCore
import CoreImage
import simd
import XLDKit

private let xldDebugCIContext = CIContext()

// MARK: - 상태 (nonisolated 콜백용)

final class TrackerState: @unchecked Sendable {
    var emaYaw: Double = 0
    var emaPitch: Double = 0
    var emaRoll: Double = 0
    var lastFrameTime: CFTimeInterval = 0
    var smoothedFPS: Double = 0
    var debugEnabled: Bool = false
    // 눈 lid 자동 캘리브레이션 (사람별)
    // -1 = 아직 관측 안 됨 (warm-up 단계)
    var eyeAspectOpen_L: Float = -1
    var eyeAspectClosed_L: Float = -1
    var eyeAspectOpen_R: Float = -1
    var eyeAspectClosed_R: Float = -1
    var eyeCalibFrames: Int = 0
    var emaLid_L: Float = 1.0
    var emaLid_R: Float = 1.0
    var lidEMAInitialized: Bool = false
    // 재캘리브레이션 트리거 (외부에서 true 로)
    var needsCalibReset: Bool = false
}

// MARK: - CameraTracker

@MainActor
final class CameraTracker: NSObject, ObservableObject {
    @Published var isRunning: Bool = false
    @Published var errorMessage: String?

    /// 기존 경로 (yaw/pitch/roll deg)
    var onAngles: ((Double, Double, Double) -> Void)?
    var onStatus: ((Bool, Double) -> Void)?
    /// 신규 경로 (skeleton)
    var onFrame: ((XLDTrackingFrame) -> Void)?
    /// 디버그 뷰용 raw landmarks
    var onRawFace: ((RawFaceLandmarks) -> Void)?
    /// 디버그 뷰용 overlay 이미지
    var onDebugImage: ((CGImage) -> Void)?

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "xld.camera", qos: .userInitiated)
    private var device: AVCaptureDevice?
    private var configuredFPS: Int = 60
    private var isStarting: Bool = false

    nonisolated let state = TrackerState()
    private let emaAlpha: Double = 0.35

    func start(targetFPS: Int = 60) {
        print("[CameraTracker] start() called, targetFPS=\(targetFPS)")
        guard !isRunning && !isStarting else { return }
        isStarting = true
        configuredFPS = targetFPS
        errorMessage = nil

        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized: beginSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    if granted { self?.beginSession() }
                    else { self?.errorMessage = "Camera access denied"; self?.isStarting = false }
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
            self.setupSession(targetFPS: fps)
            if !self.session.isRunning { self.session.startRunning() }
            Task { @MainActor in
                self.isRunning = true
                self.isStarting = false
                print("[CameraTracker] fully started")
            }
        }
    }

    private func setupSession(targetFPS: Int) {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1280x720

        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
              ?? AVCaptureDevice.default(for: .video) else {
            errorMessage = "No camera device"; return
        }
        device = dev

        do {
            let inp = try AVCaptureDeviceInput(device: dev)
            if session.canAddInput(inp) { session.addInput(inp) }
        } catch {
            errorMessage = "Input error: \(error.localizedDescription)"; return
        }

        let out = AVCaptureVideoDataOutput()
        out.alwaysDiscardsLateVideoFrames = true
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(out) { session.addOutput(out) }

        let ranges = dev.activeFormat.videoSupportedFrameRateRanges
        let supportedMax = ranges.map { $0.maxFrameRate }.max() ?? 30
        let supportedMin = ranges.map { $0.minFrameRate }.min() ?? 15
        let actualFPS = max(supportedMin, min(Double(targetFPS), supportedMax))
        do {
            try dev.lockForConfiguration()
            let dur = CMTime(value: 1, timescale: CMTimeScale(actualFPS))
            dev.activeVideoMinFrameDuration = dur
            dev.activeVideoMaxFrameDuration = dur
            dev.unlockForConfiguration()
        } catch { }
    }

    // MARK: - Feature 계산 (bbox 정규화 좌표계, top-left origin)

    nonisolated private static func clamp01(_ v: Float) -> Float {
        return max(0, min(1, v))
    }

    nonisolated private static func makeSkeleton(
        from face: VNFaceObservation,
        t: TimeInterval,
        camState: TrackerState
    ) -> (skeleton: XLDTrackingSkeleton, raw: RawFaceLandmarks?, yawDeg: Double, pitchDeg: Double, rollDeg: Double) {

        var skel = XLDTrackingSkeleton(t: t)
        let yaw = face.yaw?.doubleValue ?? 0
        let pitch = face.pitch?.doubleValue ?? 0
        let roll = face.roll?.doubleValue ?? 0
        let yawDeg = yaw * 180.0 / .pi
        let pitchDeg = pitch * 180.0 / .pi
        let rollDeg = roll * 180.0 / .pi

        let qy = simd_quatf(angle: Float(yaw),   axis: SIMD3<Float>(0, 1, 0))
        let qx = simd_quatf(angle: Float(pitch), axis: SIMD3<Float>(1, 0, 0))
        let qz = simd_quatf(angle: Float(roll),  axis: SIMD3<Float>(0, 0, 1))
        skel.head = Joint(rot: qz * qx * qy, confidence: Float(face.confidence))

        guard let lm = face.landmarks else {
            return (skel, nil, yawDeg, pitchDeg, rollDeg)
        }

        var points: [Landmark2D] = []
        var regionIndices: [FaceRegion: [Int]] = [:]

        func addRegion(_ region: VNFaceLandmarkRegion2D?, key: FaceRegion) -> [Int] {
            guard let r = region else { return [] }
            var idxs: [Int] = []
            for i in 0..<r.pointCount {
                let p = r.normalizedPoints[i]
                // Vision normalizedPoints: bbox 내 0~1, bottom-left origin
                // → top-left origin 으로 뒤집기
                idxs.append(points.count)
                points.append(Landmark2D(x: Float(p.x), y: Float(1.0 - p.y), confidence: 1))
            }
            regionIndices[key] = idxs
            return idxs
        }

        let leftEyeIdx    = addRegion(lm.leftEye,      key: .leftEye)
        let rightEyeIdx   = addRegion(lm.rightEye,     key: .rightEye)
        let leftBrowIdx   = addRegion(lm.leftEyebrow,  key: .leftBrow)
        let rightBrowIdx  = addRegion(lm.rightEyebrow, key: .rightBrow)
        let outerLipsIdx  = addRegion(lm.outerLips,    key: .outerLips)
        let innerLipsIdx  = addRegion(lm.innerLips,    key: .innerLips)
        let faceContourIdx = addRegion(lm.faceContour, key: .faceContour)

        // Pupil (Vision 은 1점)
        var pupilLIdx: Int? = nil
        var pupilRIdx: Int? = nil
        if let pl = lm.leftPupil {
            pupilLIdx = points.count
            let p = pl.normalizedPoints[0]
            points.append(Landmark2D(x: Float(p.x), y: Float(1.0 - p.y)))
        }
        if let pr = lm.rightPupil {
            pupilRIdx = points.count
            let p = pr.normalizedPoints[0]
            points.append(Landmark2D(x: Float(p.x), y: Float(1.0 - p.y)))
        }

        // ---- eyeLid: 눈 영역 종횡비 ----
        func eyeAspect(_ idxs: [Int]) -> Float {
            guard idxs.count >= 3 else { return 0 }
            let xs = idxs.map { points[$0].x }
            let ys = idxs.map { points[$0].y }
            let h = xs.max()! - xs.min()!
            let v = ys.max()! - ys.min()!
            return h > 1e-6 ? v / h : 0
        }
        let aspectL = eyeAspect(leftEyeIdx)
        let aspectR = eyeAspect(rightEyeIdx)

        // 외부 reset 요청 시 calibration 초기화
        if camState.needsCalibReset {
            camState.eyeAspectOpen_L = -1
            camState.eyeAspectClosed_L = -1
            camState.eyeAspectOpen_R = -1
            camState.eyeAspectClosed_R = -1
            camState.eyeCalibFrames = 0
            camState.lidEMAInitialized = false
            camState.needsCalibReset = false
            print("[calib] reset")
        }
        // aspect 클수록 눈 감음. 자동 캘리브레이션.

        // ── head-up 가드 (A+C, landmark 기반) ──
        // Vision face.pitch 는 양자화되어 신뢰 불가. landmark 의 mouthRatio 로 판단.
        //   정면 → mouthRatio ≈ 0.65
        //   head up → mouthRatio ↑ (>0.72)
        //   head down → mouthRatio ↓ (<0.58)
        let browAllH = leftBrowIdx + rightBrowIdx
        let browTopYH: Float = browAllH.isEmpty ? 0 : (browAllH.map { points[$0].y }.min() ?? 0)
        let faceBotYH: Float = faceContourIdx.map { points[$0].y }.max() ?? 1
        let mouthTopYH: Float = innerLipsIdx.isEmpty ? 0 : (innerLipsIdx.map { points[$0].y }.min() ?? 0)
        let vertRangeH = max(faceBotYH - browTopYH, 1e-6)
        let mouthRatioH = (mouthTopYH - browTopYH) / vertRangeH
        let pitchEst = (mouthRatioH - 0.65) * 1.5
        let headUp = abs(pitchEst) > 0.10

        // running min/max 추적 (사람별 적응)
        // open = 낮은 aspect (뜸), closed = 높은 aspect (감음)
        let aL = aspectL
        let aR = aspectR
        let alpha: Float = 0.02  // 느린 적응
        
        func updateRange(_ a: Float, open: inout Float, closed: inout Float) {
            guard a > 0.01 else { return }
            let aF = a
            if open < 0 { open = aF }       // 첫 관측값 채택
            if closed < 0 { closed = aF }   // 첫 관측값 채택
            let oDiff = aF - open
            let oRate = oDiff < 0 ? alpha : alpha * 0.3
            open += oRate * oDiff

            let cDiff = aF - closed
            let cRate = cDiff > 0 ? alpha : alpha * 0.3
            closed += cRate * cDiff

            // 최소 폭 보장 (0.10)
            if closed - open < 0.10 {
                let mid = (closed + open) * 0.5
                open = mid - 0.05
                closed = mid + 0.05
            }
        }
        if !headUp {
            updateRange(aL, open: &camState.eyeAspectOpen_L, closed: &camState.eyeAspectClosed_L)
            updateRange(aR, open: &camState.eyeAspectOpen_R, closed: &camState.eyeAspectClosed_R)
        }

        // 정규화: (closed - current) / (closed - open) → 1=뜸, 0=감음
        // 양눈 range 공유 (open/closed 를 평균으로 통합)
        let avgOpen = (camState.eyeAspectOpen_L + camState.eyeAspectOpen_R) * 0.5
        let avgClosed = (camState.eyeAspectClosed_L + camState.eyeAspectClosed_R) * 0.5
        let rangeShared = max(avgClosed - avgOpen, 0.10)
        var lidL = clamp01((camState.eyeAspectClosed_L - aspectL) / rangeShared)
        var lidR = clamp01((camState.eyeAspectClosed_R - aspectR) / rangeShared)
        // warm-up: 첫 1초(30프레임) 동안은 lid=1.0 강제 (오작동 방지)
        if camState.eyeCalibFrames < 90 || camState.eyeAspectOpen_L < 0 || camState.eyeAspectClosed_L < 0 {
            lidL = 1.0
            lidR = 1.0
        }
        if headUp {
            lidL = 1.0
            lidR = 1.0
        }
        // EMA 스무딩 (0.3 = 부드럽게)
        let lidAlpha: Float = 0.3
        if !camState.lidEMAInitialized {
            camState.emaLid_L = lidL
            camState.emaLid_R = lidR
            camState.lidEMAInitialized = true
        } else {
            camState.emaLid_L += lidAlpha * (lidL - camState.emaLid_L)
            camState.emaLid_R += lidAlpha * (lidR - camState.emaLid_R)
        }
        skel.eyeLid_L = camState.emaLid_L
        skel.eyeLid_R = camState.emaLid_R

        camState.eyeCalibFrames += 1

        // ---- iris: 동공 - 눈 중심 (눈 반폭으로 정규화) ----
        func irisOffset(pupilIdx: Int?, eyeIdx: [Int]) -> SIMD2<Float> {
            guard let pi = pupilIdx, !eyeIdx.isEmpty else { return .zero }
            let xs = eyeIdx.map { points[$0].x }
            let ys = eyeIdx.map { points[$0].y }
            let cx = xs.reduce(0, +) / Float(xs.count)
            let cy = ys.reduce(0, +) / Float(ys.count)
            let ex = (xs.max()! - xs.min()!) * 0.5
            let ey = (ys.max()! - ys.min()!) * 0.5
            let p = points[pi]
            return SIMD2<Float>(
                ex > 1e-6 ? (p.x - cx) / ex : 0,
                ey > 1e-6 ? (p.y - cy) / ey : 0
            )
        }
        skel.iris_L = irisOffset(pupilIdx: pupilLIdx, eyeIdx: leftEyeIdx)
        skel.iris_R = irisOffset(pupilIdx: pupilRIdx, eyeIdx: rightEyeIdx)

        // ---- brow: (눈 중심 y - 눈썹 y) / 눈 높이  (양수=눈썹 위) ----
        func browRel(browIdx: [Int], eyeIdx: [Int]) -> Float {
            guard !browIdx.isEmpty, !eyeIdx.isEmpty else { return 0 }
            let by = browIdx.map { points[$0].y }.reduce(0, +) / Float(browIdx.count)
            let ey = eyeIdx.map { points[$0].y }.reduce(0, +) / Float(eyeIdx.count)
            let eyeH = eyeIdx.map { points[$0].y }.max()! - eyeIdx.map { points[$0].y }.min()!
            return eyeH > 1e-6 ? (ey - by) / eyeH : 0
        }
        skel.brow_L = browRel(browIdx: leftBrowIdx,  eyeIdx: leftEyeIdx)
        skel.brow_R = browRel(browIdx: rightBrowIdx, eyeIdx: rightEyeIdx)

        // ---- mouthOpen: innerLips 수직 / outerLips 수평 ----
        if !innerLipsIdx.isEmpty, !outerLipsIdx.isEmpty {
            let iy: [Float] = innerLipsIdx.map { points[$0].y }
            let ox: [Float] = outerLipsIdx.map { points[$0].x }
            let v: Float = (iy.max() ?? 0) - (iy.min() ?? 0)
            let h: Float = (ox.max() ?? 0) - (ox.min() ?? 0)
            if h > 1e-6 {
                let ratio: Float = v / h
                skel.mouthOpen = clamp01(ratio * Float(2.5))
            } else {
                skel.mouthOpen = 0
            }
        }

        // ---- mouthWidth: outerLips 수평 크기 ----
        if !outerLipsIdx.isEmpty {
            let xs = outerLipsIdx.map { points[$0].x }
            skel.mouthWidth = xs.max()! - xs.min()!
        }

        // ---- mouthCorner: outerLips 좌/우 코너의 y - 중심 y ----
        if outerLipsIdx.count >= 9 {
            let leftCorner  = points[outerLipsIdx[0]]
            let rightCorner = points[outerLipsIdx[8]]
            let cy = outerLipsIdx.map { points[$0].y }.reduce(0, +) / Float(outerLipsIdx.count)
            let ys = outerLipsIdx.map { points[$0].y }
            let mh = ys.max()! - ys.min()!
            if mh > 1e-6 {
                skel.mouthCorner_L = (leftCorner.y  - cy) / mh
                skel.mouthCorner_R = (rightCorner.y - cy) / mh
            }
        }

        // jaw 는 일단 0 (Step 3 에서 faceContour 기반으로 채움)

        // ---- landmark 기반 연속 yaw/pitch/roll ----
        // Vision yaw/pitch는 양자화됨 (0, ±44 만 나옴). landmark로 다시 계산.

        func centroid(_ idxs: [Int]) -> SIMD2<Float>? {
            guard !idxs.isEmpty else { return nil }
            var sx: Float = 0, sy: Float = 0
            for i in idxs { sx += points[i].x; sy += points[i].y }
            return SIMD2<Float>(sx / Float(idxs.count), sy / Float(idxs.count))
        }

        var lmYaw: Double = 0
        var lmPitch: Double = 0
        var lmRoll: Double = 0

        let eyeL_c = centroid(leftEyeIdx)
        let eyeR_c = centroid(rightEyeIdx)
        let contour_c = centroid(faceContourIdx)

        if !faceContourIdx.isEmpty {
            let xs = faceContourIdx.map { points[$0].x }
            let ys = faceContourIdx.map { points[$0].y }
            let faceW = max(xs.max()! - xs.min()!, 1e-6)
            let faceH = max(ys.max()! - ys.min()!, 1e-6)
            let faceCx = (xs.max()! + xs.min()!) * 0.5
            let faceCy = (ys.max()! + ys.min()!) * 0.5

            // 코 끝: outerLips 상단 근처, 혹은 nose 인덱스가 없어서 innerLips 상단 사용
            // 더 정확히는 eye L/R 중심의 중점과 outerLips 상단 사이 중간
            var noseX: Float = faceCx
            var noseY: Float = faceCy
            if !innerLipsIdx.isEmpty {
                let iy = innerLipsIdx.map { points[$0].y }
                let ix = innerLipsIdx.map { points[$0].x }
                noseY = iy.min()!  // inner lips 상단 y
                noseX = ix.reduce(0, +) / Float(ix.count)
            }
            // pitch: 눈썹 상단 ~ 턱 하단 사이에서 입술 상단 위치
            let browAll = leftBrowIdx + rightBrowIdx
            let browTopY: Float = browAll.isEmpty
                ? (faceContourIdx.map { points[$0].y }.min() ?? 0)
                : (browAll.map { points[$0].y }.min() ?? 0)
            let faceBotY: Float = faceContourIdx.map { points[$0].y }.max() ?? 1
            let mouthTopY: Float = innerLipsIdx.isEmpty
                ? (browTopY + faceBotY) * 0.5
                : (innerLipsIdx.map { points[$0].y }.min() ?? 0)
            let vertRange = max(faceBotY - browTopY, 1e-6)
            let mouthRatio = (mouthTopY - browTopY) / vertRange
            // 사람 정면에서 입 상단은 눈썹~턱의 ~0.65 지점
            lmPitch = Double((mouthRatio - 0.65) * 1.5)
            // yaw: 코 x 편차
            let dx = (noseX - faceCx) / (faceW * 0.5)
            lmYaw = Double(dx) * 1.0   // rad-ish 근사

            if let el = eyeL_c, let er = eyeR_c {
                lmRoll = Double(atan2(er.y - el.y, er.x - el.x))
            }
        }

        // landmark 각도를 degree 로. yaw/pitch/roll 반환 교체
        let yawDegLm = lmYaw * 180.0 / .pi * 0.6   // 스케일: 과하지 않게
        let pitchDegLm = lmPitch * 180.0 / .pi * 0.6
        let rollDegLm = lmRoll * 180.0 / .pi

        // 진단
        if !faceContourIdx.isEmpty {
            let xs = faceContourIdx.map { points[$0].x }
            let ys = faceContourIdx.map { points[$0].y }
            let faceW = xs.max()! - xs.min()!
            let faceH = ys.max()! - ys.min()!
            if let el = eyeL_c, let er = eyeR_c {
                let eyeMidY = (el.y + er.y) * 0.5
                let noseY: Float = innerLipsIdx.isEmpty
                    ? 0 : innerLipsIdx.map { points[$0].y }.min()!
                let noseX: Float = innerLipsIdx.isEmpty
                    ? 0 : innerLipsIdx.map { points[$0].x }.reduce(0,+) / Float(innerLipsIdx.count)
                let faceCx = (xs.max()! + xs.min()!) * 0.5
                let dy = (noseY - eyeMidY) / max(faceH, 1e-6)
                let dx = (noseX - faceCx) / max(faceW * 0.5, 1e-6)
                let browAll2 = leftBrowIdx + rightBrowIdx
                let bTopY: Float = browAll2.isEmpty ? 0 : (browAll2.map { points[$0].y }.min() ?? 0)
                let fBotY: Float = faceContourIdx.map { points[$0].y }.max() ?? 1
                let mTopY: Float = innerLipsIdx.isEmpty ? 0 : (innerLipsIdx.map { points[$0].y }.min() ?? 0)
                let vR = max(fBotY - bTopY, 1e-6)
                let mRatio = (mTopY - bTopY) / vR
                if Int(CACurrentMediaTime() * 3) % 3 == 0 {
                    print(String(format: "[lm] bTopY=%.3f fBotY=%.3f mTopY=%.3f  mRatio=%.3f  pitchRad=%+.3f",
                                 bTopY, fBotY, mTopY, mRatio, lmPitch))
                }
            }
        }

        _ = yawDeg  // Vision 값은 참고용

        // head quaternion은 landmark 값으로
        let qy2 = simd_quatf(angle: Float(lmYaw),   axis: SIMD3<Float>(0, 1, 0))
        let qx2 = simd_quatf(angle: Float(lmPitch), axis: SIMD3<Float>(1, 0, 0))
        let qz2 = simd_quatf(angle: Float(lmRoll),  axis: SIMD3<Float>(0, 0, 1))
        skel.head = Joint(rot: qz2 * qx2 * qy2, confidence: Float(face.confidence))

        // 진단: eye aspect + lid
        if Int(CACurrentMediaTime() * 3) % 3 == 0 {
            let oL = camState.eyeAspectOpen_L
            let cL = camState.eyeAspectClosed_L
            let oR = camState.eyeAspectOpen_R
            let cR = camState.eyeAspectClosed_R
            let lidL = skel.eyeLid_L
            let lidR = skel.eyeLid_R
            print(String(format: "[eye] %@ L: a=%.3f [%.3f..%.3f] lid=%.2f | R: a=%.3f [%.3f..%.3f] lid=%.2f",
                         headUp ? "UP " : "   ",
                         aspectL, oL, cL, lidL, aspectR, oR, cR, lidR))
        }

        let raw = RawFaceLandmarks(points: points, imageSize: SIMD2<Float>(1, 1))
        // regionIndices 저장
        _ = regionIndices  // 나중에 RawFaceLandmarks 에 필드 추가 예정
        return (skel, raw, yawDegLm, pitchDegLm, rollDegLm)
    }

    nonisolated private static func drawDebugImage(
        pb: CVPixelBuffer,
        face: VNFaceObservation
    ) -> CGImage? {
        let ci = CIImage(cvPixelBuffer: pb)
        guard let baseCG = xldDebugCIContext.createCGImage(ci, from: ci.extent) else { return nil }
        let W = baseCG.width
        let H = baseCG.height
        guard let cgctx = CGContext(data: nil, width: W, height: H,
                                    bitsPerComponent: 8, bytesPerRow: 0,
                                    space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)
        else { return nil }

        cgctx.draw(baseCG, in: CGRect(x: 0, y: 0, width: W, height: H))

        let bb = face.boundingBox
        let fW = CGFloat(W), fH = CGFloat(H)

        func toCG(_ p: CGPoint) -> CGPoint {
            let nx = CGFloat(bb.minX + p.x * bb.width)
            let ny = CGFloat(bb.minY + p.y * bb.height)
            // CGContext 는 top-left 기준 → Vision bbox y(bottom-left) 뒤집기
            return CGPoint(x: (1.0 - nx) * fW, y: (1.0 - ny) * fH)
        }

        cgctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 1, alpha: 0.8))
        cgctx.setLineWidth(2)
        let bboxRect = CGRect(x: (1.0 - CGFloat(bb.maxX)) * fW,
                              y: (1.0 - CGFloat(bb.maxY)) * fH,
                              width: CGFloat(bb.width) * fW,
                              height: CGFloat(bb.height) * fH)
        cgctx.stroke(bboxRect)

        guard let lm = face.landmarks else { return cgctx.makeImage() }

        func drawRegion(_ r: VNFaceLandmarkRegion2D?, color: CGColor, radius: CGFloat = 3) {
            guard let r = r else { return }
            cgctx.setFillColor(color)
            for i in 0..<r.pointCount {
                let p = toCG(r.normalizedPoints[i])
                cgctx.fillEllipse(in: CGRect(x: p.x - radius, y: p.y - radius,
                                             width: radius * 2, height: radius * 2))
            }
        }

        // 모든 76 landmark 그리기 (region 별 색)
        drawRegion(lm.faceContour,   color: CGColor(red: 0.75, green: 0.75, blue: 0.75, alpha: 1), radius: 3)
        drawRegion(lm.leftEye,       color: CGColor(red: 0,    green: 1,    blue: 0,    alpha: 1), radius: 4)
        drawRegion(lm.rightEye,      color: CGColor(red: 0,    green: 1,    blue: 0,    alpha: 1), radius: 4)
        drawRegion(lm.leftEyebrow,   color: CGColor(red: 0.2,  green: 0.5,  blue: 1,    alpha: 1), radius: 4)
        drawRegion(lm.rightEyebrow,  color: CGColor(red: 0.2,  green: 0.5,  blue: 1,    alpha: 1), radius: 4)
        drawRegion(lm.outerLips,     color: CGColor(red: 1,    green: 0.3,  blue: 0.3,  alpha: 1), radius: 4)
        drawRegion(lm.innerLips,     color: CGColor(red: 1,    green: 0.6,  blue: 0.6,  alpha: 1), radius: 3)
        drawRegion(lm.nose,          color: CGColor(red: 1,    green: 0.85, blue: 0,    alpha: 1), radius: 3)
        drawRegion(lm.noseCrest,     color: CGColor(red: 1,    green: 0.55, blue: 0,    alpha: 1), radius: 3)
        drawRegion(lm.medianLine,    color: CGColor(red: 0.7,  green: 0.4,  blue: 1,    alpha: 1), radius: 3)

        func drawPupil(_ r: VNFaceLandmarkRegion2D?) {
            guard let r = r, r.pointCount > 0 else { return }
            let p = toCG(r.normalizedPoints[0])
            cgctx.setFillColor(CGColor(red: 1, green: 1, blue: 0, alpha: 1))
            cgctx.fillEllipse(in: CGRect(x: p.x - 7, y: p.y - 7, width: 14, height: 14))
        }
        drawPupil(lm.leftPupil)
        drawPupil(lm.rightPupil)

        // landmark 개수 표시 (좌상단 작게)
        var count = 0
        let regions: [VNFaceLandmarkRegion2D?] = [
            lm.faceContour, lm.leftEye, lm.rightEye,
            lm.leftEyebrow, lm.rightEyebrow,
            lm.outerLips, lm.innerLips,
            lm.nose, lm.noseCrest, lm.medianLine
        ]
        for r in regions { count += r?.pointCount ?? 0 }
        count += 2  // pupils

        return cgctx.makeImage()
    }
}

// MARK: - Capture Delegate

extension CameraTracker: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // VNDetectFaceLandmarksRequest: bbox + yaw/pitch/roll + 76 landmark
        let req = VNDetectFaceLandmarksRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pb,
                                            orientation: .upMirrored,
                                            options: [:])
        do { try handler.perform([req]) } catch { return }

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

        let (skel, raw, yawDeg, pitchDeg, rollDeg) = Self.makeSkeleton(from: face, t: now, camState: st)

        // EMA on angles (기존 동작 유지)
        let alpha = self.emaAlpha
        st.emaYaw   += alpha * (yawDeg   - st.emaYaw)
        st.emaPitch += alpha * (pitchDeg - st.emaPitch)
        st.emaRoll  += alpha * (rollDeg  - st.emaRoll)

        let fy = st.emaYaw
        let fp = st.emaPitch
        let fr = st.emaRoll

        var frameSkel = skel
        // EMA 적용된 각도 → quaternion 재구성
        let fYawRad = Float(fy * .pi / 180)
        let fPitchRad = Float(fp * .pi / 180)
        let fRollRad = Float(fr * .pi / 180)
        let qy = simd_quatf(angle: fYawRad,   axis: SIMD3<Float>(0, 1, 0))
        let qx = simd_quatf(angle: fPitchRad, axis: SIMD3<Float>(1, 0, 0))
        let qz = simd_quatf(angle: fRollRad,  axis: SIMD3<Float>(0, 0, 1))
        frameSkel.head = Joint(rot: qz * qx * qy, confidence: skel.head.confidence)

        let frame = XLDTrackingFrame(skeleton: frameSkel, rawFace: raw, source: "vision.face.landmarks")

        let debugCG: CGImage? = st.debugEnabled
            ? Self.drawDebugImage(pb: pb, face: face)
            : nil

        Task { @MainActor in
            self.onAngles?(fy, fp, fr)
            self.onFrame?(frame)
            if let r = raw { self.onRawFace?(r) }
            if let cg = debugCG { self.onDebugImage?(cg) }
            self.onStatus?(true, fps)
        }
    }
}
