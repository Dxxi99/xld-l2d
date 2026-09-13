import Foundation
import MetalKit
import QuartzCore
import simd
import XLDKit

final class RigRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelineState: MTLRenderPipelineState?
    var vertexBuffer: MTLBuffer?
    var indexBuffer: MTLBuffer?
    var texture: MTLTexture?
    var samplerState: MTLSamplerState?

    weak var mtkView: MTKView?
    var yawDeg: Float = 0
    var pitchDeg: Float = 0

    // depth 변위 배율 (튜닝 포인트)
    var depthFrontW: Float = 1.0   // 앞쪽 vertex 배율 (1.0 = 무효과)
    var depthBackW: Float  = 1.0   // 뒤쪽 vertex 배율

    // === Face feature (트래킹 → 캐릭터) ===
    var eyeOpen_L: Float = 1.0     // 1=뜸, 0=감음
    var eyeOpen_R: Float = 1.0
    var mouthOpen: Float = 0.0     // 0~1

    // 캐릭터 얼굴 눈/입 중심 (정규화 0~1, 캐릭터 이미지 기준)
    // meta.txt 의 auto_face_pivot_y 를 기준으로 추정, 나중에 자동 추출 가능
    var eyeL_centerN: SIMD2<Float> = SIMD2(0.38, 0.22)
    var eyeR_centerN: SIMD2<Float> = SIMD2(0.62, 0.22)
    var mouth_centerN: SIMD2<Float> = SIMD2(0.50, 0.30)
    var eye_radiusN: Float = 0.06   // 이미지 폭 대비
    var mouth_radiusN: Float = 0.08

    // 변형 강도 (튜닝)
    var eyeSquash: Float = 0.0     // 눈 감을 때 Y 압축 계수
    var mouthStretch: Float = 0.8   // 입 벌릴 때 Y 확대 계수

    // FPS 측정
    private var lastFrameTime: CFTimeInterval = 0
    private var smoothedFPS: Double = 0
    var onFPSUpdate: ((Double) -> Void)?

    let model: XLDRuntimeModel
    let sourceImageURL: URL

    init(model: XLDRuntimeModel, sourceImageURL: URL) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("Metal not available")
        }
        self.device = device
        self.commandQueue = queue
        self.model = model
        self.sourceImageURL = sourceImageURL
        super.init()
        buildPipeline()
        buildBuffers()
        buildTexture()
        buildSampler()
    }

    private func buildPipeline() {
        do {
            let opts = MTLCompileOptions()
            let lib = try device.makeLibrary(source: rigShaderSource, options: opts)
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = lib.makeFunction(name: "rig_vertex")
            desc.fragmentFunction = lib.makeFunction(name: "rig_fragment")
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].sourceRGBBlendFactor = .one
            desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.colorAttachments[0].sourceAlphaBlendFactor = .one
            desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            pipelineState = try device.makeRenderPipelineState(descriptor: desc)
            print("[RigRenderer] pipeline ok (runtime-compiled)")
        } catch {
            print("[RigRenderer] pipeline error: \(error)")
        }
    }

    private func buildBuffers() {
        let verts = model.mesh.vertices
        let bx = model.rig.basisX
        let by = model.rig.basisY
        let tris = model.mesh.triangles

        let W = Float(model.width)
        let H = Float(model.height)

        let vz = model.vertexZ
        var vertexData = [Float](repeating: 0, count: verts.count * 8)
        for i in 0..<verts.count {
            let v = verts[i]
            vertexData[i*8 + 0] = v.x
            vertexData[i*8 + 1] = v.y
            vertexData[i*8 + 2] = bx[i]
            vertexData[i*8 + 3] = by[i]
            vertexData[i*8 + 4] = v.x / W
            vertexData[i*8 + 5] = v.y / H
            vertexData[i*8 + 6] = vz[i]
            vertexData[i*8 + 7] = 0
        }
        vertexBuffer = device.makeBuffer(bytes: vertexData,
                                         length: vertexData.count * MemoryLayout<Float>.size,
                                         options: .storageModeManaged)

        var indexData = [UInt32]()
        indexData.reserveCapacity(tris.count * 3)
        for t in tris {
            indexData.append(UInt32(t.x))
            indexData.append(UInt32(t.y))
            indexData.append(UInt32(t.z))
        }
        indexBuffer = device.makeBuffer(bytes: indexData,
                                        length: indexData.count * MemoryLayout<UInt32>.size,
                                        options: .storageModeManaged)
        print("[RigRenderer] buffers: \(verts.count)v \(tris.count)t")
    }

    private func buildTexture() {
        let loader = MTKTextureLoader(device: device)
        do {
            let opts: [MTKTextureLoader.Option: Any] = [
                .origin: MTKTextureLoader.Origin.topLeft,
                .SRGB: false,
                .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue)
            ]
            texture = try loader.newTexture(URL: sourceImageURL, options: opts)
            print("[RigRenderer] texture: \(texture?.width ?? 0)x\(texture?.height ?? 0)")
        } catch {
            print("[RigRenderer] texture error: \(error)")
        }
    }

    private func buildSampler() {
        let desc = MTLSamplerDescriptor()
        desc.minFilter = .linear
        desc.magFilter = .linear
        desc.sAddressMode = .clampToEdge
        desc.tAddressMode = .clampToEdge
        samplerState = device.makeSamplerState(descriptor: desc)
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pipeline = pipelineState,
              let vb = vertexBuffer,
              let ib = indexBuffer,
              let tex = texture,
              let smp = samplerState,
              let passDesc = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: passDesc)
        else { return }

        let sclX = Float(model.width) / 1856.0
        let sclY = Float(model.height) / 2640.0
        let tuneR = model.meta.autoTuneR * 1.3
        let sx = sin(yawDeg * .pi / 180) * model.rig.rX * sclX * tuneR
        let sy = sin(pitchDeg * .pi / 180) * model.rig.rY * sclY * tuneR

        struct Uniforms {
            var viewport: SIMD2<Float>
            var imageSize: SIMD2<Float>
            var offset: SIMD2<Float>
            var scale: Float
            var pad: Float
            var headWeightBase: Float
            var headZLo: Float
            var headZHi: Float
            var lowerYStart: Float
            var lowerYEnd: Float
            var lowerFade: Float
            var yawRad: Float
            var pitchRad: Float
            var facePivotY: Float
            var faceRotScale: Float
            var facePersp: Float
            var depthFrontW: Float
            var depthBackW: Float

            var eyeLCenter: SIMD2<Float>
            var eyeRCenter: SIMD2<Float>
            var mouthCenter: SIMD2<Float>
            var eyeRadius: Float
            var mouthRadius: Float
            var eyeOpen_L: Float
            var eyeOpen_R: Float
            var mouthOpen: Float
            var eyeSquash: Float
            var mouthStretch: Float
            var pad2: Float
        }
        let W = Float(model.width), H = Float(model.height)
        let vp = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))
        let scale = min(vp.x / W, vp.y / H)
        let yawRad = yawDeg * .pi / 180
        let pitchRad = pitchDeg * .pi / 180
        var u = Uniforms(viewport: vp,
                         imageSize: SIMD2(W, H),
                         offset: SIMD2(sx, sy),
                         scale: scale,
                         pad: 0,
                         headWeightBase: 0.45,
                         headZLo: model.meta.autoHeadZLo,
                         headZHi: model.meta.autoHeadZHi,
                         lowerYStart: model.meta.autoLowerYStart,
                         lowerYEnd: model.meta.autoLowerYEnd,
                         lowerFade: 0.90,
                         yawRad: yawRad,
                         pitchRad: pitchRad,
                         facePivotY: model.meta.autoFacePivotY,
                         faceRotScale: model.meta.autoFaceRotScale,
                         facePersp: 0.07,
                         depthFrontW: depthFrontW,
                         depthBackW: depthBackW,
                         eyeLCenter: SIMD2(eyeL_centerN.x * W, eyeL_centerN.y * H),
                         eyeRCenter: SIMD2(eyeR_centerN.x * W, eyeR_centerN.y * H),
                         mouthCenter: SIMD2(mouth_centerN.x * W, mouth_centerN.y * H),
                         eyeRadius: eye_radiusN * W,
                         mouthRadius: mouth_radiusN * W,
                         eyeOpen_L: eyeOpen_L,
                         eyeOpen_R: eyeOpen_R,
                         mouthOpen: mouthOpen,
                         eyeSquash: eyeSquash,
                         mouthStretch: mouthStretch,
                         pad2: 0)

        enc.setRenderPipelineState(pipeline)
        enc.setVertexBuffer(vb, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.size, index: 1)
        enc.setFragmentTexture(tex, index: 0)
        enc.setFragmentSamplerState(smp, index: 0)
        enc.drawIndexedPrimitives(type: .triangle,
                                  indexCount: ib.length / MemoryLayout<UInt32>.size,
                                  indexType: .uint32,
                                  indexBuffer: ib,
                                  indexBufferOffset: 0)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()

        let now = CACurrentMediaTime()
        if lastFrameTime > 0 {
            let dt = now - lastFrameTime
            let inst = 1.0 / max(dt, 1e-6)
            smoothedFPS = smoothedFPS == 0 ? inst : smoothedFPS * 0.9 + inst * 0.1
            if Int(now * 4) % 4 == 0 {
                onFPSUpdate?(smoothedFPS)
            }
        }
        lastFrameTime = now
    }
}
