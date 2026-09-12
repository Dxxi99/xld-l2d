import Foundation
import Metal

public struct FullPipelineResult: Sendable {
    public let totalPixels: Int
    public let pass1Accepted: Int
    public let pass1Errors: Int
    public let errorTiles: Int
    public let pass2Accepted: Int
    public let pass2Errors: Int
    public let gpuMs: Double
}

public final class MetalPipeline: @unchecked Sendable {
    public let confThreshold: Float
    public let neighborLimit: Float
    public let edgeThreshold: Float
    public let textureThreshold: Float

    private let runtime: MetalRuntime
    private let sobelPSO: MTLComputePipelineState
    private let validatePSO: MTLComputePipelineState
    private let dilatePSO: MTLComputePipelineState
    private let reprocessPSO: MTLComputePipelineState

    private var pixelCapacity = 0
    private var bufRGBA:        MTLBuffer?
    private var bufClassID:     MTLBuffer?
    private var bufConf:        MTLBuffer?
    private var bufAccept:      MTLBuffer?
    private var bufError:       MTLBuffer?
    private var bufErrorDilate: MTLBuffer?
    private var bufClassIDOut:  MTLBuffer?
    private var bufConfOut:     MTLBuffer?
    private var bufError2:      MTLBuffer?

    public init?(confThreshold: Float = 0.5,
                 neighborLimit: Float = 0.6,
                 edgeThreshold: Float = 0.08,
                 textureThreshold: Float = 0.02) {
        guard let rt = MetalRuntime.shared,
              let f1 = rt.library.makeFunction(name: "sobel_classify"),
              let f2 = rt.library.makeFunction(name: "validate_pixels"),
              let f3 = rt.library.makeFunction(name: "dilate3x3"),
              let f4 = rt.library.makeFunction(name: "reprocess_vote")
        else { return nil }
        do {
            self.sobelPSO     = try rt.device.makeComputePipelineState(function: f1)
            self.validatePSO  = try rt.device.makeComputePipelineState(function: f2)
            self.dilatePSO    = try rt.device.makeComputePipelineState(function: f3)
            self.reprocessPSO = try rt.device.makeComputePipelineState(function: f4)
        } catch { return nil }

        self.runtime = rt
        self.confThreshold = confThreshold
        self.neighborLimit = neighborLimit
        self.edgeThreshold = edgeThreshold
        self.textureThreshold = textureThreshold
    }

    private func ensureCapacity(_ count: Int) {
        if count <= pixelCapacity, bufRGBA != nil { return }
        let cap = max(count, pixelCapacity * 2)
        let s: MTLResourceOptions = .storageModeShared
        bufRGBA        = runtime.device.makeBuffer(length: cap * 4 * MemoryLayout<Float>.size, options: s)
        bufClassID     = runtime.device.makeBuffer(length: cap, options: s)
        bufConf        = runtime.device.makeBuffer(length: cap * MemoryLayout<Float>.size, options: s)
        bufAccept      = runtime.device.makeBuffer(length: cap, options: s)
        bufError       = runtime.device.makeBuffer(length: cap, options: s)
        bufErrorDilate = runtime.device.makeBuffer(length: cap, options: s)
        bufClassIDOut  = runtime.device.makeBuffer(length: cap, options: s)
        bufConfOut     = runtime.device.makeBuffer(length: cap * MemoryLayout<Float>.size, options: s)
        bufError2      = runtime.device.makeBuffer(length: cap, options: s)
        pixelCapacity = cap
    }

    private func encodeDispatch(_ enc: MTLComputeCommandEncoder, w: Int, h: Int) {
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    public func classify(image: RasterImage) -> [PixelLabel]? {
        let w = image.width, h = image.height, count = w * h
        ensureCapacity(count)
        guard let bRGBA = bufRGBA, let bCID = bufClassID, let bConf = bufConf else { return nil }
        memcpy(bRGBA.contents(), image.rgba, count * 4 * MemoryLayout<Float>.size)
        var wu = UInt32(w), hu = UInt32(h)
        var et = edgeThreshold, tt = textureThreshold
        guard let cmd = runtime.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(sobelPSO)
        enc.setBuffer(bRGBA, offset: 0, index: 0)
        enc.setBuffer(bCID,  offset: 0, index: 1)
        enc.setBuffer(bConf, offset: 0, index: 2)
        enc.setBytes(&wu, length: 4, index: 3)
        enc.setBytes(&hu, length: 4, index: 4)
        enc.setBytes(&et, length: 4, index: 5)
        enc.setBytes(&tt, length: 4, index: 6)
        encodeDispatch(enc, w: w, h: h)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        let cidPtr = bCID.contents().bindMemory(to: UInt8.self, capacity: count)
        let confPtr = bConf.contents().bindMemory(to: Float.self, capacity: count)
        var out = [PixelLabel](repeating: PixelLabel(classID: 0, confidence: 1), count: count)
        for i in 0..<count { out[i] = PixelLabel(classID: cidPtr[i], confidence: confPtr[i]) }
        return out
    }

    public func validate(labels: [PixelLabel],
                         width: Int, height: Int) -> ValidationResult? {
        let count = width * height
        ensureCapacity(count)
        guard let bCID = bufClassID, let bConf = bufConf,
              let bAcc = bufAccept, let bErr = bufError else { return nil }
        let cidPtr = bCID.contents().bindMemory(to: UInt8.self, capacity: count)
        let confPtr = bConf.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { cidPtr[i] = labels[i].classID; confPtr[i] = labels[i].confidence }
        var wu = UInt32(width), hu = UInt32(height)
        var ct = confThreshold, nl = neighborLimit
        guard let cmd = runtime.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(validatePSO)
        enc.setBuffer(bCID,  offset: 0, index: 0)
        enc.setBuffer(bConf, offset: 0, index: 1)
        enc.setBuffer(bAcc,  offset: 0, index: 2)
        enc.setBuffer(bErr,  offset: 0, index: 3)
        enc.setBytes(&wu, length: 4, index: 4)
        enc.setBytes(&hu, length: 4, index: 5)
        enc.setBytes(&ct, length: 4, index: 6)
        enc.setBytes(&nl, length: 4, index: 7)
        encodeDispatch(enc, w: width, h: height)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        let aPtr = bAcc.contents().bindMemory(to: UInt8.self, capacity: count)
        let ePtr = bErr.contents().bindMemory(to: UInt8.self, capacity: count)
        var acc = [UInt8](repeating: 0, count: count)
        var err = [UInt8](repeating: 0, count: count)
        var aC = 0, eC = 0
        for i in 0..<count { acc[i] = aPtr[i]; err[i] = ePtr[i]
            if aPtr[i] == 1 { aC += 1 }; if ePtr[i] == 1 { eC += 1 } }
        return ValidationResult(acceptMask: acc, errorMask: err,
                                acceptedCount: aC, errorCount: eC)
    }

    public func isolateTiles(errorMask: [UInt8],
                             width: Int, height: Int,
                             tileSize: Int) -> [TileCoord]? {
        let count = width * height
        ensureCapacity(count)
        guard let bErr = bufError, let bDil = bufErrorDilate else { return nil }
        let ePtr = bErr.contents().bindMemory(to: UInt8.self, capacity: count)
        for i in 0..<count { ePtr[i] = errorMask[i] }
        var wu = UInt32(width), hu = UInt32(height)
        guard let cmd = runtime.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(dilatePSO)
        enc.setBuffer(bErr, offset: 0, index: 0)
        enc.setBuffer(bDil, offset: 0, index: 1)
        enc.setBytes(&wu, length: 4, index: 2)
        enc.setBytes(&hu, length: 4, index: 3)
        encodeDispatch(enc, w: width, h: height)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        let dPtr = bDil.contents().bindMemory(to: UInt8.self, capacity: count)
        var s = Set<TileCoord>()
        for y in 0..<height { for x in 0..<width {
            if dPtr[y * width + x] == 1 { s.insert(TileCoord(x: x/tileSize, y: y/tileSize)) } } }
        return s.sorted { $0.y == $1.y ? $0.x < $1.x : $0.y < $1.y }
    }

    public func reprocess(labels: [PixelLabel], errorMask: [UInt8],
                          width: Int, height: Int, radius: Int = 2) -> [PixelLabel]? {
        let count = width * height
        ensureCapacity(count)
        guard let bCID = bufClassID, let bConf = bufConf,
              let bErr = bufError, let bCID2 = bufClassIDOut, let bConf2 = bufConfOut
        else { return nil }
        let cidPtr = bCID.contents().bindMemory(to: UInt8.self, capacity: count)
        let confPtr = bConf.contents().bindMemory(to: Float.self, capacity: count)
        let ePtr = bErr.contents().bindMemory(to: UInt8.self, capacity: count)
        for i in 0..<count { cidPtr[i] = labels[i].classID
            confPtr[i] = labels[i].confidence; ePtr[i] = errorMask[i] }
        var wu = UInt32(width), hu = UInt32(height)
        var r = Int32(radius)
        guard let cmd = runtime.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(reprocessPSO)
        enc.setBuffer(bCID,  offset: 0, index: 0)
        enc.setBuffer(bConf, offset: 0, index: 1)
        enc.setBuffer(bErr,  offset: 0, index: 2)
        enc.setBuffer(bCID2, offset: 0, index: 3)
        enc.setBuffer(bConf2, offset: 0, index: 4)
        enc.setBytes(&wu, length: 4, index: 5)
        enc.setBytes(&hu, length: 4, index: 6)
        enc.setBytes(&r,  length: 4, index: 7)
        encodeDispatch(enc, w: width, h: height)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        let c2 = bCID2.contents().bindMemory(to: UInt8.self, capacity: count)
        let f2 = bConf2.contents().bindMemory(to: Float.self, capacity: count)
        var out = [PixelLabel](repeating: PixelLabel(classID: 0, confidence: 1), count: count)
        for i in 0..<count { out[i] = PixelLabel(classID: c2[i], confidence: f2[i]) }
        return out
    }

    public func runFullPipeline(image: RasterImage, tileSize: Int = 16) -> FullPipelineResult? {
        let w = image.width, h = image.height, count = w * h
        ensureCapacity(count)
        guard let bRGBA = bufRGBA, let bCID = bufClassID, let bConf = bufConf,
              let bAcc = bufAccept, let bErr = bufError, let bDil = bufErrorDilate,
              let bCID2 = bufClassIDOut, let bConf2 = bufConfOut, let bErr2 = bufError2
        else { return nil }

        memcpy(bRGBA.contents(), image.rgba, count * 4 * MemoryLayout<Float>.size)

        var wu = UInt32(w), hu = UInt32(h)
        var et = edgeThreshold, tt = textureThreshold
        var ct = confThreshold, nl = neighborLimit
        var r = Int32(2)

        let t0 = Date()

        guard let cmd = runtime.queue.makeCommandBuffer() else { return nil }

        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(sobelPSO)
            enc.setBuffer(bRGBA, offset: 0, index: 0)
            enc.setBuffer(bCID,  offset: 0, index: 1)
            enc.setBuffer(bConf, offset: 0, index: 2)
            enc.setBytes(&wu, length: 4, index: 3)
            enc.setBytes(&hu, length: 4, index: 4)
            enc.setBytes(&et, length: 4, index: 5)
            enc.setBytes(&tt, length: 4, index: 6)
            encodeDispatch(enc, w: w, h: h)
            enc.endEncoding()
        }

        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(validatePSO)
            enc.setBuffer(bCID,  offset: 0, index: 0)
            enc.setBuffer(bConf, offset: 0, index: 1)
            enc.setBuffer(bAcc,  offset: 0, index: 2)
            enc.setBuffer(bErr,  offset: 0, index: 3)
            enc.setBytes(&wu, length: 4, index: 4)
            enc.setBytes(&hu, length: 4, index: 5)
            enc.setBytes(&ct, length: 4, index: 6)
            enc.setBytes(&nl, length: 4, index: 7)
            encodeDispatch(enc, w: w, h: h)
            enc.endEncoding()
        }

        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(dilatePSO)
            enc.setBuffer(bErr, offset: 0, index: 0)
            enc.setBuffer(bDil, offset: 0, index: 1)
            enc.setBytes(&wu, length: 4, index: 2)
            enc.setBytes(&hu, length: 4, index: 3)
            encodeDispatch(enc, w: w, h: h)
            enc.endEncoding()
        }

        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(reprocessPSO)
            enc.setBuffer(bCID,  offset: 0, index: 0)
            enc.setBuffer(bConf, offset: 0, index: 1)
            enc.setBuffer(bErr,  offset: 0, index: 2)
            enc.setBuffer(bCID2, offset: 0, index: 3)
            enc.setBuffer(bConf2, offset: 0, index: 4)
            enc.setBytes(&wu, length: 4, index: 5)
            enc.setBytes(&hu, length: 4, index: 6)
            enc.setBytes(&r,  length: 4, index: 7)
            encodeDispatch(enc, w: w, h: h)
            enc.endEncoding()
        }

        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(validatePSO)
            enc.setBuffer(bCID2, offset: 0, index: 0)
            enc.setBuffer(bConf2, offset: 0, index: 1)
            enc.setBuffer(bAcc,  offset: 0, index: 2)
            enc.setBuffer(bErr2, offset: 0, index: 3)
            enc.setBytes(&wu, length: 4, index: 4)
            enc.setBytes(&hu, length: 4, index: 5)
            enc.setBytes(&ct, length: 4, index: 6)
            enc.setBytes(&nl, length: 4, index: 7)
            encodeDispatch(enc, w: w, h: h)
            enc.endEncoding()
        }

        cmd.commit()
        cmd.waitUntilCompleted()

        let gpuMs = Date().timeIntervalSince(t0) * 1000

        let err1P = bErr.contents().bindMemory(to: UInt8.self, capacity: count)
        let err2P = bErr2.contents().bindMemory(to: UInt8.self, capacity: count)
        let dilP  = bDil.contents().bindMemory(to: UInt8.self, capacity: count)

        var e1 = 0, e2 = 0
        for i in 0..<count { if err1P[i] == 1 { e1 += 1 }; if err2P[i] == 1 { e2 += 1 } }

        var tiles = Set<TileCoord>()
        for y in 0..<h { for x in 0..<w {
            if dilP[y * w + x] == 1 { tiles.insert(TileCoord(x: x/tileSize, y: y/tileSize)) } } }

        return FullPipelineResult(
            totalPixels: count,
            pass1Accepted: count - e1,
            pass1Errors: e1,
            errorTiles: tiles.count,
            pass2Accepted: count - e2,
            pass2Errors: e2,
            gpuMs: gpuMs
        )
    }
}
