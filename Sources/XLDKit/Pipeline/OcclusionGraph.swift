import Foundation

public struct OcclusionGraph: Sendable {
    public let width: Int
    public let height: Int
    public let orderedLayers: [LayerID]
    public let layerZ: [LayerID: Int]
    public let pixelOwnerRank: [Int16]
}

public final class OcclusionGraphBuilder: @unchecked Sendable {
    public static let defaultZ: [LayerID: Int] = [
        .background: 0,
        .hairBack:   10,
        .body:       20,
        .neck:       25,
        .clothes:    30,
        .armL:       35,
        .armR:       36,
        .head:       38,
        .face:       40,
        .mouth:      50,
        .eyeL:       51,
        .eyeR:       52,
        .eyebrowL:   53,
        .eyebrowR:   54,
        .hairFront:  60,
    ]

    public init() {}

    public func build(layerMasks: [LayerID: [UInt8]],
                      width: Int, height: Int) -> OcclusionGraph {
        let z = OcclusionGraphBuilder.defaultZ

        let present = layerMasks.keys.filter { key in
            guard let m = layerMasks[key] else { return false }
            return m.contains(1)
        }
        let ordered = present.sorted { (z[$0] ?? 0) < (z[$1] ?? 0) }

        var rank: [LayerID: Int] = [:]
        for (i, l) in ordered.enumerated() { rank[l] = i }

        let total = width * height
        var owner = [Int16](repeating: -1, count: total)
        for l in ordered {
            guard let mask = layerMasks[l], mask.count == total else { continue }
            let r = Int16(rank[l] ?? 0)
            for i in 0..<total where mask[i] == 1 {
                owner[i] = r
            }
        }

        return OcclusionGraph(width: width, height: height,
                              orderedLayers: ordered,
                              layerZ: z,
                              pixelOwnerRank: owner)
    }
}
