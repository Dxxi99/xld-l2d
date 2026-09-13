import Foundation
import simd

// ─────────────────────────────────────────
// MARK: - Raw Landmarks (원본 보존)
// ─────────────────────────────────────────

public struct Landmark2D: Sendable {
    public var x: Float
    public var y: Float
    public var confidence: Float
    public init(x: Float, y: Float, confidence: Float = 1) {
        self.x = x; self.y = y; self.confidence = confidence
    }
}

public enum FaceRegion: String, Sendable, CaseIterable {
    case leftEye, rightEye
    case leftBrow, rightBrow
    case outerLips, innerLips
    case nose, faceContour
}

public struct RawFaceLandmarks: Sendable {
    public var points: [Landmark2D]
    public var imageSize: SIMD2<Float>
    public init(points: [Landmark2D], imageSize: SIMD2<Float>) {
        self.points = points
        self.imageSize = imageSize
    }
}

// ─────────────────────────────────────────
// MARK: - Joint (부모 기준 로컬 회전)
// ─────────────────────────────────────────

public struct Joint: Sendable {
    public var rot: simd_quatf
    public var confidence: Float
    public init(rot: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                confidence: Float = 0) {
        self.rot = rot
        self.confidence = confidence
    }
    public static let identity = Joint()
}

// ─────────────────────────────────────────
// MARK: - XLD Tracking Skeleton (사람 무관)
// ─────────────────────────────────────────

public struct XLDTrackingSkeleton: Sendable {

    public var t: TimeInterval

    // Head: neck 기준 로컬 회전
    public var head: Joint = .identity

    // Face
    public var jaw: Joint = .identity
    public var eyeLid_L: Float = 1.0
    public var eyeLid_R: Float = 1.0
    public var iris_L: SIMD2<Float> = .zero
    public var iris_R: SIMD2<Float> = .zero
    public var brow_L: Float = 0
    public var brow_R: Float = 0
    public var mouthOpen: Float = 0
    public var mouthWidth: Float = 0
    public var mouthCorner_L: Float = 0
    public var mouthCorner_R: Float = 0

    public init(t: TimeInterval) { self.t = t }

    public static func empty(t: TimeInterval = 0) -> Self { .init(t: t) }
}

// ─────────────────────────────────────────
// MARK: - Frame (skeleton + raw + meta)
// ─────────────────────────────────────────

public struct XLDTrackingFrame: Sendable {
    public var skeleton: XLDTrackingSkeleton
    public var rawFace: RawFaceLandmarks?
    public var source: String

    public init(skeleton: XLDTrackingSkeleton,
                rawFace: RawFaceLandmarks? = nil,
                source: String = "unknown") {
        self.skeleton = skeleton
        self.rawFace = rawFace
        self.source = source
    }
}
