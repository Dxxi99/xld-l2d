import Foundation
import Accelerate

/// Canonical fusion: SCC + topological order + single pass z adjustment.
/// Mirrors tools/scripts/layer_promotion_v2.py exactly.
public struct CanonicalFusion {
    public static let eps: Float = Float(0.005)

    public struct Constraint {
        public let stem: Int32
        public let ca: Int32
        public let cb: Int32
        public init(stem: Int32, ca: Int32, cb: Int32) {
            self.stem = stem; self.ca = ca; self.cb = cb
        }
    }

    public struct Result {
        public let z: [Float]
        public let topoOrder: [Int32]
        public let sccLabel: [Int32]
        public let sccCount: Int
        public let violations: Int
    }

    public static func fuse(
        segZ: [Float],
        nSeg: Int,
        constraints: [Constraint]
    ) -> Result {
        let n = nSeg + 1
        var eu = [Int32](); eu.reserveCapacity(constraints.count * 2)
        var ev = [Int32](); ev.reserveCapacity(constraints.count * 2)
        for c in constraints {
            eu.append(c.ca); ev.append(c.stem)
            eu.append(c.cb); ev.append(c.stem)
        }
        let m = eu.count

        let (sccLabel, nScc) = tarjanSCC(n: n, from: eu, to: ev, edgeCount: m)

        var inDeg = [Int32](repeating: 0, count: nScc)
        var outAdj = [[Int32]](repeating: [], count: nScc)
        for i in 0..<m {
            let a = sccLabel[Int(eu[i])]
            let b = sccLabel[Int(ev[i])]
            if a == b { continue }
            outAdj[Int(a)].append(b)
            inDeg[Int(b)] += 1
        }

        var queue = [Int32]()
        for i in 0..<nScc where inDeg[i] == 0 { queue.append(Int32(i)) }
        var head = 0
        var topo = [Int32](); topo.reserveCapacity(nScc)
        while head < queue.count {
            let u = queue[head]; head += 1
            topo.append(u)
            for v in outAdj[Int(u)] {
                inDeg[Int(v)] -= 1
                if inDeg[Int(v)] == 0 { queue.append(v) }
            }
        }

        var sccMembers = [[Int32]](repeating: [], count: nScc)
        for s in 1...nSeg {
            sccMembers[Int(sccLabel[s])].append(Int32(s))
        }

        var z = segZ
        for uScc in topo {
            let members = sccMembers[Int(uScc)]
            if members.isEmpty { continue }
            var zU = z[Int(members[0])]
            for mm in members {
                if z[Int(mm)] < zU { zU = z[Int(mm)] }
            }
            let target = zU - eps
            for vScc in outAdj[Int(uScc)] {
                for mv in sccMembers[Int(vScc)] {
                    if z[Int(mv)] > target { z[Int(mv)] = target }
                }
            }
        }

        var viol = 0
        for c in constraints {
            let zs = z[Int(c.stem)]
            let zca = z[Int(c.ca)]
            let zcb = z[Int(c.cb)]
            let mn = min(zca, zcb)
            if zs > mn + 1e-6 { viol += 1 }
        }

        return Result(z: z, topoOrder: topo, sccLabel: sccLabel,
                      sccCount: nScc, violations: viol)
    }

    // MARK: - Tarjan SCC (iterative)

    private static func tarjanSCC(
        n: Int, from: [Int32], to: [Int32], edgeCount: Int
    ) -> ([Int32], Int) {
        var head = [Int32](repeating: -1, count: n)
        var next = [Int32](repeating: -1, count: edgeCount)
        for i in 0..<edgeCount {
            let u = Int(from[i])
            next[i] = head[u]
            head[u] = Int32(i)
        }

        var index   = [Int32](repeating: -1, count: n)
        var lowlink = [Int32](repeating:  0, count: n)
        var onStack = [Bool](repeating: false, count: n)
        var sccLabel = [Int32](repeating: -1, count: n)
        var stack = [Int32]()
        var idx: Int32 = 0
        var sccCount: Int32 = 0

        struct Frame {
            var v: Int32
            var edge: Int32
        }
        var callStack = [Frame]()

        for start in 0..<n where index[start] < 0 {
            callStack.append(Frame(v: Int32(start), edge: head[start]))
            index[start] = idx; lowlink[start] = idx; idx += 1
            stack.append(Int32(start)); onStack[start] = true

            while !callStack.isEmpty {
                let top = callStack.count - 1
                let v = Int(callStack[top].v)

                if callStack[top].edge != -1 {
                    let e = Int(callStack[top].edge)
                    let w = Int(to[e])
                    callStack[top].edge = next[e]

                    if index[w] < 0 {
                        index[w] = idx; lowlink[w] = idx; idx += 1
                        stack.append(Int32(w)); onStack[w] = true
                        callStack.append(Frame(v: Int32(w), edge: head[w]))
                    } else if onStack[w] {
                        if index[w] < lowlink[v] { lowlink[v] = index[w] }
                    }
                } else {
                    callStack.removeLast()
                    if let parentFrame = callStack.last {
                        let p = Int(parentFrame.v)
                        if lowlink[v] < lowlink[p] { lowlink[p] = lowlink[v] }
                    }
                    if lowlink[v] == index[v] {
                        while true {
                            let w = stack.removeLast()
                            onStack[Int(w)] = false
                            sccLabel[Int(w)] = sccCount
                            if Int(w) == v { break }
                        }
                        sccCount += 1
                    }
                }
            }
        }

        return (sccLabel, Int(sccCount))
    }
}
