import Foundation

public struct ValidationResult: Sendable {
    public let acceptMask: [UInt8]
    public let errorMask:  [UInt8]
    public let acceptedCount: Int
    public let errorCount: Int
}

public final class PixelValidator: @unchecked Sendable {
    public let confidenceThreshold: Float
    public let neighborDisagreeLimit: Float

    public init(confidenceThreshold: Float = 0.5,
                neighborDisagreeLimit: Float = 0.6) {
        self.confidenceThreshold = confidenceThreshold
        self.neighborDisagreeLimit = neighborDisagreeLimit
    }

    public func validate(labels: [PixelLabel],
                         width: Int, height: Int) -> ValidationResult {
        var accept = [UInt8](repeating: 0, count: width * height)
        var error  = [UInt8](repeating: 0, count: width * height)
        var accCount = 0, errCount = 0

        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let l = labels[idx]

                var ok = l.confidence >= confidenceThreshold

                if ok {
                    var disagree = 0
                    var total = 0
                    for dy in -1...1 {
                        for dx in -1...1 {
                            if dx == 0 && dy == 0 { continue }
                            let nx = x + dx, ny = y + dy
                            if nx < 0 || ny < 0 || nx >= width || ny >= height { continue }
                            let nl = labels[ny * width + nx]
                            total += 1
                            if abs(Int(nl.classID) - Int(l.classID)) >= 2 { disagree += 1 }
                        }
                    }
                    if total > 0,
                       Float(disagree) / Float(total) > neighborDisagreeLimit {
                        ok = false
                    }
                }

                if ok { accept[idx] = 1; accCount += 1 }
                else  { error[idx]  = 1; errCount += 1 }
            }
        }
        return ValidationResult(acceptMask: accept,
                                errorMask: error,
                                acceptedCount: accCount,
                                errorCount: errCount)
    }
}
