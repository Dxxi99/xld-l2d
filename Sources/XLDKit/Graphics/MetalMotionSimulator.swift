import Foundation
import Metal

public final class MetalMotionSimulator: @unchecked Sendable {
    private let runtime: MetalRuntime
    private let poseRankPSO: MTLComputePipelineState
    private let comparePSO:  MTLComputePipelineState
    private let unionPSO:    MTLComputePipelineState

    // persistent buffers
    private var width = 0
    private var height = 0
    private var pixelCount = 0

    private var bufMasks:    MTLBuffer?     // [nLayers][W*H] uchar
    private var bufDxs:      MTLBuffer?     // [nLayers] int32
    private var bufDys:      MTLBuffer?     // [nLayers] int32
    private var bufRanks:    MTLBuffer?     // [nLayers] int16
    private var bufOrig:     MTLBuffer?     // [W*H] int16
    private var bufPoseRank: MTLBuffer?     // [W*H] int16
    private var bufDisocc:   MTLBuffer?     // [W*H] uchar
    private var bufLifetime: MTLBuffer?     // [W*H] uchar

    public init?() {
        guard let rt = MetalRuntime.shared,
              let f1 = rt.library.makeFunction(name: "pose_top_rank"),
              let f2 = rt.library.makeFunction(name: "compare_rank"),
              let f3 = rt.library.makeFunction(name: "union_mask")
        else { return nil }
        do {
            self.poseRankPSO = try rt.device.makeComputePipelineState(function: f1)
            self.comparePSO  = try rt.device.makeComputePipelineState(function: f2)
            self.unionPSO    = try rt.device.makeComputePipelineState(function: f3)
        } catch { return nil }
        self.runtime = rt
    }

    private func ensure(_ count: Int) {
        if count <= pixelCount, bufMasks != nil { return }
        let cap = max(count, pixelCount * 2)
        let s: MTLResourceOptions = .storageModeShared
        bufMasks    = nil // reallocate below when nLayers known
        bufPoseRank = runtime.device.makeBuffer(length: cap * 2, options: s)
        bufDisocc   = runtime.device.makeBuffer(length: cap, options: s)
        bufLifetime = runtime.device.makeBuffer(length: cap, options: s)
        bufOrig     = runtime.device.makeBuffer(length: cap * 2, options: s)
        pixelCount = cap
    }

    public func simulate(sequence: MotionSequence,
                         layerMasks: [LayerID: [UInt8]],
                         layerZ: [LayerID: Int],
                         originalTopRank: [Int16],
                         orderedLayers: [LayerID],
                         width: Int, height: Int) -> LifetimeResult? {
        let W = width, H = height, total = W * H
        let nLayers = orderedLayers.count
        if nLayers == 0 { return nil }

        ensure(total)
        guard let bufPoseRank = bufPoseRank,
              let bufDisocc   = bufDisocc,
              let bufLifetime = bufLifetime,
              let bufOrig     = bufOrig
        else { return nil }

        // masks
        let maskBytes = nLayers * total
        if bufMasks == nil || (bufMasks?.length ?? 0) < maskBytes {
            bufMasks = runtime.device.makeBuffer(length: maskBytes,
                                                  options: .storageModeShared)
        }
        guard let bufMasks = bufMasks else { return nil }

        let maskPtr = bufMasks.contents().bindMemory(to: UInt8.self, capacity: maskBytes)
        for (i, layer) in orderedLayers.enumerated() {
            guard let m = layerMasks[layer] else { continue }
            let base = i * total
            for j in 0..<total { maskPtr[base + j] = m[j] }
        }

        // original top rank
        let origPtr = bufOrig.contents().bindMemory(to: Int16.self, capacity: total)
        for i in 0..<total { origPtr[i] = originalTopRank[i] }

        // small per-layer buffers
        let smallBytes = nLayers * MemoryLayout<Int32>.size
        if bufDxs == nil || (bufDxs?.length ?? 0) < smallBytes {
            bufDxs = runtime.device.makeBuffer(length: smallBytes, options: .storageModeShared)
            bufDys = runtime.device.makeBuffer(length: smallBytes, options: .storageModeShared)
        }
        let rankBytes = nLayers * MemoryLayout<Int16>.size
        if bufRanks == nil || (bufRanks?.length ?? 0) < rankBytes {
            bufRanks = runtime.device.makeBuffer(length: rankBytes, options: .storageModeShared)
        }
        guard let bufDxs = bufDxs, let bufDys = bufDys, let bufRanks = bufRanks else { return nil }

        // ranks are 0..nLayers-1
        let rankPtr = bufRanks.contents().bindMemory(to: Int16.self, capacity: nLayers)
        for i in 0..<nLayers { rankPtr[i] = Int16(i) }

        // lifetime accumulate
        let ltPtr = bufLifetime.contents().bindMemory(to: UInt8.self, capacity: total)
        for i in 0..<total { ltPtr[i] = 0 }

        var wu = UInt32(W), hu = UInt32(H), nu = UInt32(nLayers), tu = UInt32(total)

        // Build one big command buffer
        guard let cmd = runtime.queue.makeCommandBuffer() else { return nil }

        // First pass: write per-pose top rank, compare, union
        for pose in sequence.poses {
            // fill dx/dy for this pose
            let dxPtr = bufDxs.contents().bindMemory(to: Int32.self, capacity: nLayers)
            let dyPtr = bufDys.contents().bindMemory(to: Int32.self, capacity: nLayers)
            for i in 0..<nLayers {
                let layer = orderedLayers[i]
                if let move = pose.keys.first(where: { $0.layer == layer }) {
                    dxPtr[i] = Int32(move.dx)
                    dyPtr[i] = Int32(move.dy)
                } else {
                    dxPtr[i] = 0
                    dyPtr[i] = 0
                }
            }

            // pose_top_rank
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(poseRankPSO)
                enc.setBuffer(bufMasks,    offset: 0, index: 0)
                enc.setBuffer(bufDxs,      offset: 0, index: 1)
                enc.setBuffer(bufDys,      offset: 0, index: 2)
                enc.setBuffer(bufRanks,    offset: 0, index: 3)
                enc.setBuffer(bufPoseRank, offset: 0, index: 4)
                enc.setBytes(&wu, length: 4, index: 5)
                enc.setBytes(&hu, length: 4, index: 6)
                enc.setBytes(&nu, length: 4, index: 7)
                enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                enc.endEncoding()
            }

            // compare_rank
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(comparePSO)
                enc.setBuffer(bufOrig,     offset: 0, index: 0)
                enc.setBuffer(bufPoseRank, offset: 0, index: 1)
                enc.setBuffer(bufDisocc,   offset: 0, index: 2)
                enc.setBytes(&tu, length: 4, index: 3)
                enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }

            // union into lifetime
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(unionPSO)
                enc.setBuffer(bufDisocc,   offset: 0, index: 0)
                enc.setBuffer(bufLifetime, offset: 0, index: 1)
                enc.setBytes(&tu, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
        }

        cmd.commit()
        cmd.waitUntilCompleted()

        // readback per-pose stats + lifetime
        let ltOut = Array(UnsafeBufferPointer(start: ltPtr, count: total))
        var lifetimeCount = 0
        for i in 0..<total where ltOut[i] == 1 { lifetimeCount += 1 }

        // We didn't preserve per-pose disocclusion masks; recompute stats by
        // reading bufDisocc — but it's overwritten each pose. For stats, do a
        // second pass reading each pose's disocclusion.
        // To keep it simple, per-pose stats come out as nil-safe approximations:
        var poseResults: [PoseResult] = []
        // Second pass with readback per pose (only if stats needed)
        // Skip for now — caller only uses lifetime for pipeline. Provide basic pose count.

        // Recompute per-pose via a small second command buffer with wait per pose?
        // We'll approximate: iterate poses again, dispatching and reading.
        for pose in sequence.poses {
            let dxPtr = bufDxs.contents().bindMemory(to: Int32.self, capacity: nLayers)
            let dyPtr = bufDys.contents().bindMemory(to: Int32.self, capacity: nLayers)
            for i in 0..<nLayers {
                let layer = orderedLayers[i]
                if let move = pose.keys.first(where: { $0.layer == layer }) {
                    dxPtr[i] = Int32(move.dx)
                    dyPtr[i] = Int32(move.dy)
                } else {
                    dxPtr[i] = 0; dyPtr[i] = 0
                }
            }
            guard let cmd2 = runtime.queue.makeCommandBuffer() else { continue }
            if let enc = cmd2.makeComputeCommandEncoder() {
                enc.setComputePipelineState(poseRankPSO)
                enc.setBuffer(bufMasks,    offset: 0, index: 0)
                enc.setBuffer(bufDxs,      offset: 0, index: 1)
                enc.setBuffer(bufDys,      offset: 0, index: 2)
                enc.setBuffer(bufRanks,    offset: 0, index: 3)
                enc.setBuffer(bufPoseRank, offset: 0, index: 4)
                enc.setBytes(&wu, length: 4, index: 5)
                enc.setBytes(&hu, length: 4, index: 6)
                enc.setBytes(&nu, length: 4, index: 7)
                enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                enc.endEncoding()
            }
            if let enc = cmd2.makeComputeCommandEncoder() {
                enc.setComputePipelineState(comparePSO)
                enc.setBuffer(bufOrig,     offset: 0, index: 0)
                enc.setBuffer(bufPoseRank, offset: 0, index: 1)
                enc.setBuffer(bufDisocc,   offset: 0, index: 2)
                enc.setBytes(&tu, length: 4, index: 3)
                enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }
            cmd2.commit()
            cmd2.waitUntilCompleted()

            let dPtr = bufDisocc.contents().bindMemory(to: UInt8.self, capacity: total)
            var dCount = 0, unmasked = 0, topChanged = 0
            for i in 0..<total {
                if dPtr[i] == 1 {
                    dCount += 1
                    if origPtr[i] >= 0 && bufPoseRank.contents().bindMemory(to: Int16.self,
                                                                            capacity: total)[i] < 0 {
                        unmasked += 1
                    } else {
                        topChanged += 1
                    }
                }
            }
            var mask = [UInt8](repeating: 0, count: total)
            for i in 0..<total { mask[i] = dPtr[i] }
            poseResults.append(PoseResult(name: pose.name,
                                          disocclusionCount: dCount,
                                          topChangedCount: topChanged,
                                          unmaskedCount: unmasked,
                                          disocclusionMask: mask))
        }

        return LifetimeResult(width: W, height: H,
                              totalPixels: total,
                              sequenceName: sequence.name,
                              poseResults: poseResults,
                              lifetimeDisocclusion: ltOut,
                              lifetimeCount: lifetimeCount)
    }
}
