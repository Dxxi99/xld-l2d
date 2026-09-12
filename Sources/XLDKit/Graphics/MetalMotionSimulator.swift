import Foundation
import Metal

public final class MetalMotionSimulator: @unchecked Sendable {
    private let runtime: MetalRuntime
    private let poseRankPSO: MTLComputePipelineState
    private let comparePSO:  MTLComputePipelineState

    private var pixelCapacity = 0
    private var poseCapacity = 0

    private var bufMasks:         MTLBuffer?
    private var bufDxs:           MTLBuffer?
    private var bufDys:           MTLBuffer?
    private var bufRanks:         MTLBuffer?
    private var bufOrig:          MTLBuffer?
    private var bufPoseRankAll:   MTLBuffer?
    private var bufPoseDisoccAll: MTLBuffer?

    public init?() {
        guard let rt = MetalRuntime.shared,
              let f1 = rt.library.makeFunction(name: "pose_top_rank_at"),
              let f2 = rt.library.makeFunction(name: "compare_rank_at")
        else { return nil }
        do {
            self.poseRankPSO = try rt.device.makeComputePipelineState(function: f1)
            self.comparePSO  = try rt.device.makeComputePipelineState(function: f2)
        } catch { return nil }
        self.runtime = rt
    }

    private func ensurePixel(_ count: Int) {
        if count <= pixelCapacity, bufOrig != nil { return }
        let cap = max(count, pixelCapacity * 2)
        let s: MTLResourceOptions = .storageModeShared
        bufOrig = runtime.device.makeBuffer(length: cap * MemoryLayout<Int16>.size, options: s)
        pixelCapacity = cap
    }

    private func ensurePose(_ nPoses: Int, total: Int) {
        if nPoses <= poseCapacity, bufPoseDisoccAll != nil { return }
        let cap = max(nPoses, poseCapacity * 2, 1)
        let s: MTLResourceOptions = .storageModeShared
        bufPoseRankAll   = runtime.device.makeBuffer(
            length: cap * total * MemoryLayout<Int16>.size, options: s)
        bufPoseDisoccAll = runtime.device.makeBuffer(
            length: cap * total, options: s)
        poseCapacity = cap
    }

    public func simulate(sequence: MotionSequence,
                         layerMasks: [LayerID: [UInt8]],
                         layerZ: [LayerID: Int],
                         originalTopRank: [Int16],
                         orderedLayers: [LayerID],
                         width: Int, height: Int) -> LifetimeResult? {
        let W = width, H = height, total = W * H
        let nLayers = orderedLayers.count
        let nPoses  = sequence.poses.count
        if nLayers == 0 || nPoses == 0 { return nil }

        ensurePixel(total)
        ensurePose(nPoses, total: total)

        guard let bufOrig     = bufOrig,
              let bufPoseRank = bufPoseRankAll,
              let bufPoses    = bufPoseDisoccAll
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
            let base = i * total
            if let m = layerMasks[layer] {
                for j in 0..<total { maskPtr[base + j] = m[j] }
            } else {
                for j in 0..<total { maskPtr[base + j] = 0 }
            }
        }

        // orig
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

        let rankPtr = bufRanks.contents().bindMemory(to: Int16.self, capacity: nLayers)
        for i in 0..<nLayers { rankPtr[i] = Int16(i) }

        var wu = UInt32(W), hu = UInt32(H), nu = UInt32(nLayers), tu = UInt32(total)

        // Zero disocclusion output first (CPU, avoids Metal hazards)
        let dOutPtr0 = bufPoses.contents().bindMemory(to: UInt8.self, capacity: nPoses * total)
        for i in 0..<(nPoses * total) { dOutPtr0[i] = 0 }

        // One command buffer per pose to guarantee ordering via waitUntilCompleted
        for (poseIdx, pose) in sequence.poses.enumerated() {
            // fill dx/dy
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

            var rankElemOffset = UInt32(poseIdx * total)
            var disoccElemOffset = UInt32(poseIdx * total)

            guard let cmd = runtime.queue.makeCommandBuffer() else { return nil }

            // 1) pose_top_rank_at -> bufPoseRank[poseIdx * total ..]
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(poseRankPSO)
                enc.setBuffer(bufMasks,     offset: 0, index: 0)
                enc.setBuffer(bufDxs,       offset: 0, index: 1)
                enc.setBuffer(bufDys,       offset: 0, index: 2)
                enc.setBuffer(bufRanks,     offset: 0, index: 3)
                enc.setBuffer(bufPoseRank,  offset: 0, index: 4)
                enc.setBytes(&wu, length: 4, index: 5)
                enc.setBytes(&hu, length: 4, index: 6)
                enc.setBytes(&nu, length: 4, index: 7)
                enc.setBytes(&rankElemOffset, length: 4, index: 8)
                enc.dispatchThreads(MTLSize(width: W, height: H, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.endEncoding()
            }

            // 2) compare_rank_at -> bufPoses[poseIdx * total ..]
            if let enc = cmd.makeComputeCommandEncoder() {
                enc.setComputePipelineState(comparePSO)
                enc.setBuffer(bufOrig,     offset: 0, index: 0)
                enc.setBuffer(bufPoseRank, offset: 0, index: 1)
                enc.setBuffer(bufPoses,    offset: 0, index: 2)
                enc.setBytes(&tu, length: 4, index: 3)
                enc.setBytes(&rankElemOffset, length: 4, index: 4)
                enc.setBytes(&disoccElemOffset, length: 4, index: 5)
                enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
                enc.endEncoding()
            }

            cmd.commit()
            cmd.waitUntilCompleted()
        }

        // readback + union
        let allPtr = bufPoses.contents().bindMemory(to: UInt8.self, capacity: nPoses * total)
        var lifetime = [UInt8](repeating: 0, count: total)
        var lifetimeCount = 0

        var poseResults: [PoseResult] = []
        for (poseIdx, pose) in sequence.poses.enumerated() {
            let base = poseIdx * total
            var disocc = 0, topChanged = 0, unmasked = 0
            var mask = [UInt8](repeating: 0, count: total)
            for i in 0..<total {
                let v = allPtr[base + i]
                if v == 0 { continue }
                mask[i] = 1
                disocc += 1
                if v == 1 { topChanged += 1 } else if v == 2 { unmasked += 1 }
                if lifetime[i] == 0 { lifetime[i] = 1; lifetimeCount += 1 }
            }
            poseResults.append(PoseResult(name: pose.name,
                                          disocclusionCount: disocc,
                                          topChangedCount: topChanged,
                                          unmaskedCount: unmasked,
                                          disocclusionMask: mask))
        }

        return LifetimeResult(width: W, height: H,
                              totalPixels: total,
                              sequenceName: sequence.name,
                              poseResults: poseResults,
                              lifetimeDisocclusion: lifetime,
                              lifetimeCount: lifetimeCount)
    }
}
