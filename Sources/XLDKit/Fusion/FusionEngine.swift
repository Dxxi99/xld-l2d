import Foundation

public final class FusionEngine {

    public struct Config {
        public var regionQuantiles: [Float] = [15,30,45,60,75,90]
        public var regionMinSize: Int = 800
        public var tMargin: Float = 0.03
        public var tLambda: Float = 0.5
        public var priorMu: Float = 0.02
        public var maxIter: Int = 300
        public init() {}
    }

    public struct Result {
        public var regionLabels: [Int32]
        public var regionZ: [Float]
        public var segLabels: [Int32]
        public var segZ: [Float]
        public var tJunctions: [(y: Int, x: Int, stemDir: Int)]
        public var iterations: Int
        public var violations: Int
    }

    public let width: Int
    public let height: Int
    public let config: Config

    public init(width: Int, height: Int, config: Config = .init()) {
        self.width = width
        self.height = height
        self.config = config
    }

    public func run(
        rgb: [Float],
        depth: [Float],
        charMask: [UInt8]
    ) -> Result {
        let N = width * height

        var cd = [Float]()
        cd.reserveCapacity(N)
        for i in 0..<N where charMask[i] != 0 { cd.append(depth[i]) }
        let qs = quantiles(cd, config.regionQuantiles)
        print("[Fusion] quantiles:", qs)

        var bucket = [Int8](repeating: -1, count: N)
        for i in 0..<N where charMask[i] != 0 {
            var b = 0
            for q in qs where depth[i] > q { b += 1 }
            bucket[i] = Int8(b)
        }

        var regionLabels = [Int32](repeating: 0, count: N)
        var regionSum: [Double] = [0]
        var nextRegion: Int32 = 0
        for b in 0..<7 {
            var visited = [Bool](repeating: false, count: N)
            for i in 0..<N {
                if visited[i] || bucket[i] != Int8(b) || charMask[i] == 0 { continue }
                var comp: [Int] = []
                var q = [i]; visited[i] = true
                while let p = q.popLast() {
                    comp.append(p)
                    let py = p / width, px = p % width
                    for (dy, dx) in [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)] {
                        let ny = py + dy, nx = px + dx
                        if ny<0||ny>=height||nx<0||nx>=width { continue }
                        let ni = ny*width+nx
                        if visited[ni] { continue }
                        if bucket[ni] != Int8(b) || charMask[ni] == 0 { continue }
                        visited[ni] = true
                        q.append(ni)
                    }
                }
                if comp.count < config.regionMinSize { continue }
                nextRegion += 1
                var sum: Double = 0
                for p in comp {
                    regionLabels[p] = nextRegion
                    sum += Double(depth[p])
                }
                regionSum.append(sum / Double(comp.count))
            }
        }
        print("[Fusion] regions:", nextRegion)

        // fill unassigned
        var unassigned = [Int]()
        for i in 0..<N where charMask[i] != 0 && regionLabels[i] == 0 {
            unassigned.append(i)
        }
        for _ in 0..<20 where !unassigned.isEmpty {
            var changed = 0
            for i in unassigned {
                let py = i / width, px = i % width
                var found: Int32 = 0
                for (dy, dx) in [(0,1),(-1,0),(0,-1),(1,0)] {
                    let ny = py+dy, nx = px+dx
                    if ny<0||ny>=height||nx<0||nx>=width { continue }
                    let v = regionLabels[ny*width+nx]
                    if v > 0 { found = v; break }
                }
                if found > 0 { regionLabels[i] = found; changed += 1 }
            }
            if changed == 0 { break }
            unassigned = unassigned.filter { regionLabels[$0] == 0 }
        }

        // ---- region z (normalized) ----
        var regionZ = [Float](repeating: 0, count: max(1, Int(nextRegion)+1))
        if nextRegion > 0 {
            var rmin = Float.greatestFiniteMagnitude
            var rmax = -Float.greatestFiniteMagnitude
            for s in 1...Int(nextRegion) {
                let v = Float(regionSum[s])
                if v < rmin { rmin = v }
                if v > rmax { rmax = v }
            }
            for s in 1...Int(nextRegion) {
                regionZ[s] = (Float(regionSum[s]) - rmin) / max(1e-6, rmax - rmin)
            }
            print("[Fusion] region depth range:", rmin, "..", rmax)
        } else {
            print("[Fusion] no regions (min size too large?)")
        }

        let segLabels = [Int32](repeating: 0, count: N)
        let segZ = [Float](repeating: 0, count: 1)

        return Result(
            regionLabels: regionLabels,
            regionZ: regionZ,
            segLabels: segLabels,
            segZ: segZ,
            tJunctions: [],
            iterations: 0,
            violations: 0
        )
    }

    private func quantiles(_ arr: [Float], _ qs: [Float]) -> [Float] {
        guard !arr.isEmpty else { return qs.map { _ in 0 } }
        let sorted = arr.sorted()
        let n = sorted.count
        return qs.map { q in
            let idx = min(n-1, max(0, Int(Float(n) * q / 100.0)))
            return sorted[idx]
        }
    }
}
