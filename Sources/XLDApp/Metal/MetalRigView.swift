import SwiftUI
import MetalKit
import XLDKit

struct MetalRigView: NSViewRepresentable {
    let model: XLDRuntimeModel
    let yawDeg: Double
    let pitchDeg: Double
    let sourceImageURL: URL
    var onFPS: ((Double) -> Void)?

    init(model: XLDRuntimeModel,
         yawDeg: Double,
         pitchDeg: Double,
         sourceImageURL: URL,
         onFPS: ((Double) -> Void)? = nil) {
        self.model = model
        self.yawDeg = yawDeg
        self.pitchDeg = pitchDeg
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
    }
}
