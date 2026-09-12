import Foundation

public enum PyramidCC {
    public static func label(mask: [UInt8],
                             width: Int, height: Int,
                             factor: Int,
                             minSizeFullRes: Int) -> ComponentLabeling {
        let ds = Downsampler.downsampleMajority(mask: mask, width: width,
                                                height: height, factor: factor)
        let minDS = max(1, minSizeFullRes / (factor * factor))
        let cc = ConnectedComponents(minSize: minDS)
        let dsLabel = cc.label(mask: ds.mask, width: ds.width, height: ds.height)

        let total = width * height
        var fullIDs = [Int32](repeating: -1, count: total)
        for y in 0..<height {
            let dy = min(y / factor, ds.height - 1)
            let dsRow = dy * ds.width
            let row = y * width
            for x in 0..<width {
                let dx = min(x / factor, ds.width - 1)
                fullIDs[row + x] = dsLabel.componentID[dsRow + dx]
            }
        }

        var seen: [Int32: Int32] = [:]
        var sizes: [Int] = []
        var bboxes: [(x0: Int, y0: Int, x1: Int, y1: Int)] = []
        var sumX: [Int] = []
        var sumY: [Int] = []
        var nextID: Int32 = 0

        for i in 0..<total {
            let dsID = fullIDs[i]
            if dsID < 0 { continue }
            let newID: Int32
            if let e = seen[dsID] { newID = e }
            else {
                newID = nextID; seen[dsID] = newID; nextID += 1
                sizes.append(0)
                bboxes.append((Int.max, Int.max, Int.min, Int.min))
                sumX.append(0); sumY.append(0)
            }
            fullIDs[i] = newID
            let id = Int(newID)
            sizes[id] += 1
            let x = i % width, y = i / width
            if x < bboxes[id].x0 { bboxes[id].x0 = x }
            if y < bboxes[id].y0 { bboxes[id].y0 = y }
            if x > bboxes[id].x1 { bboxes[id].x1 = x }
            if y > bboxes[id].y1 { bboxes[id].y1 = y }
            sumX[id] += x; sumY[id] += y
        }

        return ComponentLabeling(
            componentID: fullIDs,
            componentCount: Int(nextID),
            componentSizes: sizes,
            componentBBoxes: bboxes,
            componentSumX: sumX,
            componentSumY: sumY
        )
    }
}
