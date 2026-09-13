import SwiftUI
import MetalKit
import XLDKit

struct MetalRigView: NSViewRepresentable {
    let model: XLDRuntimeModel
    let yawDeg: Double
    let pitchDeg: Double
    let eyeOpen_L: Double
    let eyeOpen_R: Double
    let mouthOpen: Double
    let eyeL_X: Double
    let eyeL_Y: Double
    let eyeR_X: Double
    let eyeR_Y: Double
    let mouth_X: Double
    let mouth_Y: Double
    let eyeR_N: Double
    let mouthR_N: Double
    let sourceImageURL: URL
    var onFPS: ((Double) -> Void)?

    init(model: XLDRuntimeModel,
         yawDeg: Double,
         pitchDeg: Double,
         eyeOpen_L: Double,
         eyeOpen_R: Double,
         mouthOpen: Double,
         eyeL_X: Double, eyeL_Y: Double,
         eyeR_X: Double, eyeR_Y: Double,
         mouth_X: Double, mouth_Y: Double,
         eyeR_N: Double, mouthR_N: Double,
         sourceImageURL: URL,
         onFPS: ((Double) -> Void)? = nil) {
        self.model = model
        self.yawDeg = yawDeg
        self.pitchDeg = pitchDeg
        self.eyeOpen_L = eyeOpen_L
        self.eyeOpen_R = eyeOpen_R
        self.mouthOpen = mouthOpen
        self.eyeL_X = eyeL_X; self.eyeL_Y = eyeL_Y
        self.eyeR_X = eyeR_X; self.eyeR_Y = eyeR_Y
        self.mouth_X = mouth_X; self.mouth_Y = mouth_Y
        self.eyeR_N = eyeR_N; self.mouthR_N = mouthR_N
        self.sourceImageURL = sourceImageURL
        self.onFPS = onFPS
    }

    func makeCoordinator() -> RigRenderer {
        let r = RigRenderer(model: model, sourceImageURL: sourceImageURL)
        r.onFPSUpdate = onFPS
        return r
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = context.coordinator.device
        view.delegate = context.coordinator
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColorMake(0.06, 0.06, 0.08, 1.0)
        view.preferredFramesPerSecond = 60
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        context.coordinator.mtkView = view
        return view
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.yawDeg = Float(yawDeg)
        context.coordinator.pitchDeg = Float(pitchDeg)
        context.coordinator.eyeOpen_L = Float(eyeOpen_L)
        context.coordinator.eyeOpen_R = Float(eyeOpen_R)
        context.coordinator.mouthOpen = Float(mouthOpen)
        context.coordinator.eyeL_centerN = SIMD2(Float(eyeL_X), Float(eyeL_Y))
        context.coordinator.eyeR_centerN = SIMD2(Float(eyeR_X), Float(eyeR_Y))
        context.coordinator.mouth_centerN = SIMD2(Float(mouth_X), Float(mouth_Y))
        context.coordinator.eye_radiusN = Float(eyeR_N)
        context.coordinator.mouth_radiusN = Float(mouthR_N)
    }
}
