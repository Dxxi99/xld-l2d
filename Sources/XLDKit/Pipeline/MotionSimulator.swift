import Foundation

public struct MotionKey: Sendable {
    public let layer: LayerID
    public let dx: Int
    public let dy: Int
    public init(_ layer: LayerID, dx: Int, dy: Int) {
        self.layer = layer; self.dx = dx; self.dy = dy
    }
}

public struct MotionPose: Sendable {
    public let name: String
    public let keys: [MotionKey]
    public init(_ name: String, _ keys: [MotionKey]) {
        self.name = name; self.keys = keys
    }
}

public struct MotionSequence: Sendable {
    public let name: String
    public let poses: [MotionPose]
    public init(_ name: String, _ poses: [MotionPose]) {
        self.name = name; self.poses = poses
    }
}

public struct PoseResult: Sendable {
    public let name: String
    public let disocclusionCount: Int
    public let topChangedCount: Int
    public let unmaskedCount: Int
    public let disocclusionMask: [UInt8]
}

public struct LifetimeResult: Sendable {
    public let width: Int
    public let height: Int
    public let totalPixels: Int
    public let sequenceName: String
    public let poseResults: [PoseResult]
    public let lifetimeDisocclusion: [UInt8]
    public let lifetimeCount: Int
}

public final class MotionSimulator: @unchecked Sendable {
    public init() {}

    public func simulate(sequence: MotionSequence,
                         layerMasks: [LayerID: [UInt8]],
                         layerZ: [LayerID: Int],
                         originalTopRank: [Int16],
                         orderedLayers: [LayerID],
                         width: Int, height: Int) -> LifetimeResult {
        let total = width * height

        let rankOf: [LayerID: Int] = {
            var m: [LayerID: Int] = [:]
            for (i, l) in orderedLayers.enumerated() { m[l] = i }
            return m
        }()

        var lifetime = [UInt8](repeating: 0, count: total)
        var lifetimeCount = 0
        var poseResults: [PoseResult] = []

        for pose in sequence.poses {
            // 1. build "pose top rank" map by layering each mask in z-order
            var poseRank = [Int16](repeating: -1, count: total)

            // sort layers by z ascending so later writes win
            let layersByZ = layerMasks.keys.sorted {
                (layerZ[$0] ?? 0) < (layerZ[$1] ?? 0)
            }

            for layer in layersByZ {
                guard let base = layerMasks[layer] else { continue }
                let rank = Int16(rankOf[layer] ?? 0)

                let move = pose.keys.first(where: { $0.layer == layer })
                let dx = move?.dx ?? 0
                let dy = move?.dy ?? 0

                for y in 0..<height {
                    for x in 0..<width {
                        let srcIdx = y * width + x
                        if base[srcIdx] == 0 { continue }
                        let nx = x + dx
                        let ny = y + dy
                        if nx < 0 || ny < 0 || nx >= width || ny >= height { continue }
                        poseRank[ny * width + nx] = rank
                    }
                }
            }

            // 2. compare against original top
            var dis = [UInt8](repeating: 0, count: total)
            var disCount = 0
            var topChanged = 0
            var unmasked = 0
            for i in 0..<total {
                let orig = originalTopRank[i]
                let now  = poseRank[i]
                if orig < 0 { continue }
                if now < 0 {
                    // was covered, now nothing = true disocclusion
                    dis[i] = 1; disCount += 1; unmasked += 1
                } else if orig != now {
                    // top layer changed = different content needed
                    dis[i] = 1; disCount += 1; topChanged += 1
                }
            }

            // 3. union into lifetime
            for i in 0..<total where dis[i] == 1 {
                if lifetime[i] == 0 { lifetimeCount += 1 }
                lifetime[i] = 1
            }

            poseResults.append(PoseResult(
                name: pose.name,
                disocclusionCount: disCount,
                topChangedCount: topChanged,
                unmaskedCount: unmasked,
                disocclusionMask: dis
            ))
        }

        return LifetimeResult(
            width: width, height: height,
            totalPixels: total,
            sequenceName: sequence.name,
            poseResults: poseResults,
            lifetimeDisocclusion: lifetime,
            lifetimeCount: lifetimeCount
        )
    }
}

public enum DefaultMotionSequences {
    public static func vtuber() -> MotionSequence {
        MotionSequence("vtuber-basic", [
            MotionPose("idle", []),

            MotionPose("blink_L", [
                MotionKey(.eyeL, dx: 0, dy: 3),
            ]),
            MotionPose("blink_R", [
                MotionKey(.eyeR, dx: 0, dy: 3),
            ]),
            MotionPose("blink_both", [
                MotionKey(.eyeL, dx: 0, dy: 3),
                MotionKey(.eyeR, dx: 0, dy: 3),
            ]),

            MotionPose("talk_open", [
                MotionKey(.mouth, dx: 0, dy: -3),
            ]),
            MotionPose("talk_wide", [
                MotionKey(.mouth, dx: 0, dy: -3),
                MotionKey(.mouth, dx: -2, dy: 0),
            ]),

            MotionPose("turn_left", [
                MotionKey(.face,      dx: -12, dy: 0),
                MotionKey(.eyeL,      dx: -12, dy: 0),
                MotionKey(.eyeR,      dx: -12, dy: 0),
                MotionKey(.mouth,     dx: -12, dy: 0),
                MotionKey(.hairFront, dx: -8,  dy: 0),
            ]),
            MotionPose("turn_right", [
                MotionKey(.face,      dx: 12, dy: 0),
                MotionKey(.eyeL,      dx: 12, dy: 0),
                MotionKey(.eyeR,      dx: 12, dy: 0),
                MotionKey(.mouth,     dx: 12, dy: 0),
                MotionKey(.hairFront, dx: 8,  dy: 0),
            ]),

            MotionPose("nod_down", [
                MotionKey(.face,    dx: 0, dy: -8),
                MotionKey(.eyeL,    dx: 0, dy: -8),
                MotionKey(.eyeR,    dx: 0, dy: -8),
                MotionKey(.mouth,   dx: 0, dy: -8),
                MotionKey(.hairFront, dx: 0, dy: -4),
            ]),
            MotionPose("nod_up", [
                MotionKey(.face,    dx: 0, dy: 8),
                MotionKey(.eyeL,    dx: 0, dy: 8),
                MotionKey(.eyeR,    dx: 0, dy: 8),
                MotionKey(.mouth,   dx: 0, dy: 8),
                MotionKey(.hairFront, dx: 0, dy: 6),
            ]),

            MotionPose("sway_left", [
                MotionKey(.hairFront, dx: -18, dy: 2),
            ]),
            MotionPose("sway_right", [
                MotionKey(.hairFront, dx: 18, dy: 2),
            ]),
        ])
    }
}
