import Foundation

public struct ComponentLabeling: Sendable {
    public let componentID: [Int32]
    public let componentCount: Int
    public let componentSizes: [Int]
    public let componentBBoxes: [(x0: Int, y0: Int, x1: Int, y1: Int)]
    public let componentSumX: [Int]
    public let componentSumY: [Int]
}

public final class ConnectedComponents: @unchecked Sendable {
    public let minSize: Int
    public init(minSize: Int = 16) { self.minSize = minSize }

    public func label(mask: [UInt8], width: Int, height: Int) -> ComponentLabeling {
        let count = width * height
        var parent: [Int32] = []
        parent.reserveCapacity(1024)
        var labels = [Int32](repeating: -1, count: count)

        @inline(__always) func find(_ x: Int32) -> Int32 {
            var r = x
            while parent[Int(r)] != r { r = parent[Int(r)] }
            var c = x
            while parent[Int(c)] != r {
                let n = parent[Int(c)]; parent[Int(c)] = r; c = n
            }
            return r
        }

        @inline(__always) func union(_ a: Int32, _ b: Int32) {
            let ra = find(a), rb = find(b)
            if ra == rb { return }
            if ra < rb { parent[Int(rb)] = ra } else { parent[Int(ra)] = rb }
        }

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                if mask[idx] == 0 { continue }
                let left: Int32 = x > 0 ? labels[idx - 1] : -1
                let up:   Int32 = y > 0 ? labels[idx - width] : -1
                if left < 0 && up < 0 {
                    let nl = Int32(parent.count); parent.append(nl); labels[idx] = nl
                } else if left >= 0 && up < 0 {
                    labels[idx] = find(left)
                } else if left < 0 && up >= 0 {
                    labels[idx] = find(up)
                } else {
                    let fl = find(left), fu = find(up)
                    labels[idx] = min(fl, fu)
                    if fl != fu { union(fl, fu) }
                }
            }
        }

        var rootToNew: [Int32: Int32] = [:]
        var nextID: Int32 = 0
        var sizes: [Int] = []
        var bboxes: [(x0: Int, y0: Int, x1: Int, y1: Int)] = []
        var sumX: [Int] = []
        var sumY: [Int] = []

        for i in 0..<count {
            if labels[i] < 0 { continue }
            let r = find(labels[i])
            let newID: Int32
            if let e = rootToNew[r] {
                newID = e
            } else {
                newID = nextID; rootToNew[r] = newID; nextID += 1
                sizes.append(0)
                bboxes.append((Int.max, Int.max, Int.min, Int.min))
                sumX.append(0); sumY.append(0)
            }
            labels[i] = newID
            let id = Int(newID)
            sizes[id] += 1
            let x = i % width, y = i / width
            if x < bboxes[id].x0 { bboxes[id].x0 = x }
            if y < bboxes[id].y0 { bboxes[id].y0 = y }
            if x > bboxes[id].x1 { bboxes[id].x1 = x }
            if y > bboxes[id].y1 { bboxes[id].y1 = y }
            sumX[id] += x; sumY[id] += y
        }

        // Single-pass small-component removal
        var isSmall = [UInt8](repeating: 0, count: sizes.count)
        for i in 0..<sizes.count where sizes[i] < minSize { isSmall[i] = 1 }
        for i in 0..<count {
            let l = labels[i]
            if l >= 0 && isSmall[Int(l)] == 1 { labels[i] = -1 }
        }

        return ComponentLabeling(
            componentID: labels,
            componentCount: Int(nextID),
            componentSizes: sizes,
            componentBBoxes: bboxes,
            componentSumX: sumX,
            componentSumY: sumY
        )
    }
}
