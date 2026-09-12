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

    // Vertex layout: position(2) + basis(2) + uv(2) = 6 floats
    private func buildBuffers() {
        let verts = model.mesh.vertices
        let bx = model.rig.basisX
        let by = model.rig.basisY
        let tris = model.mesh.triangles

        let W = Float(model.width)
        let H = Float(model.height)

        var vertexData = [Float](repeating: 0, count: verts.count * 6)
        for i in 0..<verts.count {
            let v = verts[i]
            vertexData[i*6 + 0] = v.x
            vertexData[i*6 + 1] = v.y
            vertexData[i*6 + 2] = bx[i]
            vertexData[i*6 + 3] = by[i]
            vertexData[i*6 + 4] = v.x / W
            vertexData[i*6 + 5] = v.y / H
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

        // uniforms
        let sx = sin(yawDeg * .pi / 180) * model.rig.rX
        let sy = sin(pitchDeg * .pi / 180) * model.rig.rY

        struct Uniforms {
            var viewport: SIMD2<Float>
            var imageSize: SIMD2<Float>
            var offset: SIMD2<Float>
            var scale: Float
            var pad: Float
        }
        let W = Float(model.width), H = Float(model.height)
        let vp = SIMD2<Float>(Float(view.drawableSize.width), Float(view.drawableSize.height))
        let scale = min(vp.x / W, vp.y / H)
        var u = Uniforms(viewport: vp,
                         imageSize: SIMD2(W, H),
                         offset: SIMD2(sx, sy),
                         scale: scale,
                         pad: 0)

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

        // FPS 측정
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
