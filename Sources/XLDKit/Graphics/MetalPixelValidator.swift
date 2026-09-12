import Foundation
import Metal

public final class MetalPixelValidator: @unchecked Sendable {
    public let confThreshold: Float
    public let neighborLimit: Float

    private let runtime: MetalRuntime
    private let validatePSO: MTLComputePipelineState
    private let dilatePSO: MTLComputePipelineState

    public init?(confThreshold: Float = 0.5,
                 neighborLimit: Float = 0.6) {
        guard let rt = MetalRuntime.shared,
              let vfn = rt.library.makeFunction(name: "validate_pixels"),
              let dfn = rt.library.makeFunction(name: "dilate3x3")
        else { return nil }
        do {
            self.validatePSO = try rt.device.makeComputePipelineState(function: vfn)
            self.dilatePSO   = try rt.device.makeComputePipelineState(function: dfn)
        } catch { return nil }
        self.runtime = rt
        self.confThreshold = confThreshold
        self.neighborLimit = neighborLimit
    }

    public func validate(labels: [PixelLabel],
                         width: Int, height: Int) -> ValidationResult? {
        let count = width * height
        precondition(labels.count == count)

        var classIDs = [UInt8](repeating: 0, count: count)
        var confs    = [Float](repeating: 0, count: count)
        for i in 0..<count {
            classIDs[i] = labels[i].classID
            confs[i]    = labels[i].confidence
        }

        guard let inClassID = runtime.device.makeBuffer(bytes: classIDs,
                                                        length: count,
                                                        options: .storageModeShared),
              let inConf    = runtime.device.makeBuffer(bytes: confs,
                                                        length: count * MemoryLayout<Float>.size,
                                                        options: .storageModeShared),
              let outAccept = runtime.device.makeBuffer(length: count,
                                                        options: .storageModeShared),
              let outError  = runtime.device.makeBuffer(length: count,
                                                        options: .storageModeShared)
        else { return nil }

        var w = UInt32(width), h = UInt32(height)
        var ct = confThreshold
        var nl = neighborLimit

        guard let cmd = runtime.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(validatePSO)
        enc.setBuffer(inClassID, offset: 0, index: 0)
        enc.setBuffer(inConf,    offset: 0, index: 1)
        enc.setBuffer(outAccept, offset: 0, index: 2)
        enc.setBuffer(outError,  offset: 0, index: 3)
        enc.setBytes(&w, length: 4, index: 4)
        enc.setBytes(&h, length: 4, index: 5)
        enc.setBytes(&ct, length: 4, index: 6)
        enc.setBytes(&nl, length: 4, index: 7)

        let tpg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: width, height: height, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let acceptPtr = outAccept.contents().bindMemory(to: UInt8.self, capacity: count)
        let errorPtr  = outError.contents().bindMemory(to: UInt8.self, capacity: count)
        let acceptBuf = Array(UnsafeBufferPointer(start: acceptPtr, count: count))
        let errorBuf  = Array(UnsafeBufferPointer(start: errorPtr,  count: count))
        let acceptedCount = acceptBuf.reduce(0) { $0 + Int($1) }
        let errorCount    = errorBuf.reduce(0)  { $0 + Int($1) }

        return ValidationResult(acceptMask: acceptBuf,
                                errorMask: errorBuf,
                                acceptedCount: acceptedCount,
                                errorCount: errorCount)
    }

    public func dilate(errorMask: [UInt8],
                       width: Int, height: Int) -> [UInt8]? {
        let count = width * height
        precondition(errorMask.count == count)

        guard let inBuf  = runtime.device.makeBuffer(bytes: errorMask,
                                                     length: count,
                                                     options: .storageModeShared),
              let outBuf = runtime.device.makeBuffer(length: count,
                                                     options: .storageModeShared)
        else { return nil }

        var w = UInt32(width), h = UInt32(height)

        guard let cmd = runtime.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(dilatePSO)
        enc.setBuffer(inBuf,  offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&w, length: 4, index: 2)
        enc.setBytes(&h, length: 4, index: 3)

        let tpg = MTLSize(width: 16, height: 16, depth: 1)
        let grid = MTLSize(width: width, height: height, depth: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tpg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let ptr = outBuf.contents().bindMemory(to: UInt8.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    public func isolateTiles(errorMask: [UInt8],
                             width: Int, height: Int,
                             tileSize: Int) -> [TileCoord] {
        guard let dilated = dilate(errorMask: errorMask,
                                   width: width, height: height) else { return [] }
        var set = Set<TileCoord>()
        for y in 0..<height {
            for x in 0..<width {
                if dilated[y * width + x] == 1 {
                    set.insert(TileCoord(x: x / tileSize, y: y / tileSize))
                }
            }
        }
        return set.sorted { $0.y == $1.y ? $0.x < $1.x : $0.y < $1.y }
    }
}
