import Foundation
import CoreGraphics
import ImageIO
import XLDKit

let args = Array(CommandLine.arguments.dropFirst())

func usage() {
    print("""
    XLD L2D — \(XLDKit.version)

    usage:
      xld version
      xld pipeline
      xld make-sample <out.png>
      xld run  <image.png> [--cpu|--gpu]
      xld bench <image.png>
    """)
}

func cmdVersion() {
    print("XLD L2D — \(XLDKit.version)")
    print("Pipeline version: \(XLDKit.pipelineVersion)")
    if let rt = MetalRuntime.shared {
        print("metal device: \(rt.device.name)")
    } else {
        print("metal device: unavailable")
    }
}

func cmdPipeline() {
    let pipeline = Pipeline()
    print("Pipeline stages:")
    for (i, s) in pipeline.stages.enumerated() {
        print(String(format: "  %2d. %@", i + 1, s.rawValue))
    }
    print("")
    print("Running (placeholder)...")
    let ran = pipeline.run()
    print("Done. \(ran.count) stages scheduled.")
}

func cmdMakeSample(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    try SampleImage.generate(to: url, width: 256, height: 256)
    print("sample written: \(url.path)")
}

func cmdRun(_ path: String, mode: String) throws {
    let url = URL(fileURLWithPath: path)
    let outDir = url.deletingLastPathComponent().appendingPathComponent("out")
    try? FileManager.default.createDirectory(at: outDir,
                                             withIntermediateDirectories: true)
    let t0 = Date()

    print("== XLD Run ==")
    print("input: \(url.lastPathComponent)")
    print("mode:  \(mode)")

    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    let total = W * H
    print(String(format: "image: %d x %d  (%d px)", W, H, total))

    let extractor = TileExtractor(tileSize: 16, overlap: 4)
    let plan = extractor.plan(width: W, height: H)
    print(String(format: "context tiles: %d  (size=%d, overlap=%d)",
                 plan.count, plan.tileSize, plan.overlap))

    let tClass = Date()
    let classifier = PerPixelClassifier()
    let labels1 = classifier.classify(img)
    let dtClass = Date().timeIntervalSince(tClass)

    let cpu = PixelValidator()
    let metal = MetalPixelValidator()
    let useGPU = (mode == "gpu") && metal != nil

    func runValidate(_ labels: [PixelLabel]) -> (ValidationResult, TimeInterval) {
        let t = Date()
        let r: ValidationResult
        if useGPU, let m = metal, let g = m.validate(labels: labels, width: W, height: H) {
            r = g
        } else {
            r = cpu.validate(labels: labels, width: W, height: H)
        }
        return (r, Date().timeIntervalSince(t))
    }

    let (v1, dtVal1) = runValidate(labels1)

    let tIso = Date()
    let isolator = ErrorIsolator(tileSize: 16, dilationRadius: 1)
    let etiles = isolator.isolate(errorMask: v1.errorMask, width: W, height: H)
    let dtIso = Date().timeIntervalSince(tIso)

    let tRep = Date()
    let reprocessor = ErrorReprocessor(voteRadius: 2)
    let rep = reprocessor.reprocess(labels: labels1,
                                    errorMask: v1.errorMask,
                                    width: W, height: H)
    let dtRep = Date().timeIntervalSince(tRep)

    let (v2, dtVal2) = runValidate(rep.labels)

    let cache = TileCache()
    for tile in etiles.grid {
        let key = cache.makeKey(tileBytes: [UInt8(tile.x & 0xff), UInt8(tile.y & 0xff)],
                                modelID: "per_pixel_classifier",
                                modelVersion: "v0")
        _ = cache.get(key)
        cache.put(key, Data([1]))
    }
    let stats = cache.snapshotStats()

    let labelsURL    = outDir.appendingPathComponent("labels.png")
    let errBeforeURL = outDir.appendingPathComponent("error_before.png")
    let errAfterURL  = outDir.appendingPathComponent("error_after.png")

    try DiagnosticWriter.writeLabels(labels1, width: W, height: H, url: labelsURL)
    try DiagnosticWriter.writeErrorMask(v1.errorMask, width: W, height: H, url: errBeforeURL)
    try DiagnosticWriter.writeErrorMask(v2.errorMask, width: W, height: H, url: errAfterURL)

    func pct(_ n: Int) -> String {
        return String(format: "%.2f%%", Float(n) / Float(total) * 100.0)
    }

    print("")
    print("== Pass 1: Validate ==  [\(useGPU ? "GPU" : "CPU")]")
    print(String(format: "  accepted      : %d  (%@)", v1.acceptedCount, pct(v1.acceptedCount)))
    print(String(format: "  error pixels  : %d  (%@)", v1.errorCount,    pct(v1.errorCount)))

    print("")
    print("== Error Isolation ==")
    print(String(format: "  error tiles   : %d  (tile=%d)", etiles.count, etiles.tileSize))

    print("")
    print("== Pass 2: Reprocess ==")
    print(String(format: "  pixels visited: %d", rep.pixelsVisited))
    print(String(format: "  pixels changed: %d", rep.pixelsChanged))
    print(String(format: "  pixels fixed  : %d", rep.pixelsFixed))

    print("")
    print("== Pass 2: Re-Validate ==  [\(useGPU ? "GPU" : "CPU")]")
    print(String(format: "  accepted      : %d  (%@)", v2.acceptedCount, pct(v2.acceptedCount)))
    print(String(format: "  error pixels  : %d  (%@)", v2.errorCount,    pct(v2.errorCount)))
    print(String(format: "  improvement   : -%d error pixels",
                 max(0, v1.errorCount - v2.errorCount)))

    print("")
    print("== Tile Cache ==")
    print(String(format: "  hits / misses : %d / %d", stats.hits, stats.misses))
    print(String(format: "  stores        : %d", stats.stores))

    print("")
    print("== Timing ==")
    print(String(format: "  classify      : %.1f ms", dtClass * 1000))
    print(String(format: "  validate #1   : %.1f ms", dtVal1  * 1000))
    print(String(format: "  isolate       : %.1f ms", dtIso   * 1000))
    print(String(format: "  reprocess     : %.1f ms", dtRep   * 1000))
    print(String(format: "  validate #2   : %.1f ms", dtVal2  * 1000))
    print(String(format: "  total         : %.1f ms", Date().timeIntervalSince(t0) * 1000))
}

func cmdBench(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("== XLD Bench ==")
    print("image: \(W) x \(H)  (\(W * H) px)")

    let classifier = PerPixelClassifier()
    let labels = classifier.classify(img)

    let cpu = PixelValidator()
    let metal = MetalPixelValidator()

    let t0 = Date()
    _ = cpu.validate(labels: labels, width: W, height: H)
    let dtCPU = Date().timeIntervalSince(t0)
    print(String(format: "cpu validate   : %.1f ms", dtCPU * 1000))

    if let m = metal {
        let t1 = Date()
        _ = m.validate(labels: labels, width: W, height: H)
        let dtGPU = Date().timeIntervalSince(t1)
        print(String(format: "gpu validate   : %.1f ms", dtGPU * 1000))
        let speedup = dtCPU / max(dtGPU, 0.0001)
        print(String(format: "speedup        : %.1fx", speedup))
    } else {
        print("gpu validate   : unavailable")
    }
}

func cmdBenchGPU(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("== XLD Bench-GPU ==")
    print("image: \(W) x \(H)  (\(W * H) px)")

    let cpuClass = PerPixelClassifier()
    let cpuVal   = PixelValidator()
    let cpuRep   = ErrorReprocessor()
    let cpuIso   = ErrorIsolator()

    var t = Date()
    let labelsCPU = cpuClass.classify(img)
    let dtClassCPU = Date().timeIntervalSince(t)

    t = Date()
    let v1CPU = cpuVal.validate(labels: labelsCPU, width: W, height: H)
    let dtValCPU = Date().timeIntervalSince(t)

    t = Date()
    _ = cpuIso.isolate(errorMask: v1CPU.errorMask, width: W, height: H)
    let dtIsoCPU = Date().timeIntervalSince(t)

    t = Date()
    let repCPU = cpuRep.reprocess(labels: labelsCPU, errorMask: v1CPU.errorMask,
                                  width: W, height: H)
    let dtRepCPU = Date().timeIntervalSince(t)

    print("")
    print("-- CPU --")
    print(String(format: "  classify    : %.1f ms", dtClassCPU * 1000))
    print(String(format: "  validate    : %.1f ms", dtValCPU * 1000))
    print(String(format: "  isolate     : %.1f ms", dtIsoCPU * 1000))
    print(String(format: "  reprocess   : %.1f ms", dtRepCPU * 1000))

    guard let gpu = MetalPipeline() else {
        print("gpu unavailable"); return
    }

    t = Date()
    guard let labelsGPU = gpu.classify(image: img) else {
        print("gpu classify failed"); return
    }
    let dtClassGPU = Date().timeIntervalSince(t)

    t = Date()
    guard let v1GPU = gpu.validate(labels: labelsGPU, width: W, height: H) else {
        print("gpu validate failed"); return
    }
    let dtValGPU = Date().timeIntervalSince(t)

    t = Date()
    _ = gpu.isolateTiles(errorMask: v1GPU.errorMask, width: W, height: H, tileSize: 16)
    let dtIsoGPU = Date().timeIntervalSince(t)

    t = Date()
    guard let repGPU = gpu.reprocess(labels: labelsGPU, errorMask: v1GPU.errorMask,
                                     width: W, height: H) else {
        print("gpu reprocess failed"); return
    }
    let dtRepGPU = Date().timeIntervalSince(t)

    print("")
    print("-- GPU --")
    print(String(format: "  classify    : %.1f ms", dtClassGPU * 1000))
    print(String(format: "  validate    : %.1f ms", dtValGPU * 1000))
    print(String(format: "  isolate     : %.1f ms", dtIsoGPU * 1000))
    print(String(format: "  reprocess   : %.1f ms", dtRepGPU * 1000))

    print("")
    print("-- Agreement --")
    var classDiff = 0
    for i in 0..<(W * H) where labelsCPU[i].classID != labelsGPU[i].classID { classDiff += 1 }
    print("  classify   mismatches: \(classDiff)")

    print("  validate   errors CPU=\(v1CPU.errorCount) GPU=\(v1GPU.errorCount)")

    var repDiff = 0
    for i in 0..<(W * H) where repCPU.labels[i].classID != repGPU[i].classID { repDiff += 1 }
    print("  reprocess  mismatches: \(repDiff)")

    print("")
    print("-- Speedup --")
    print(String(format: "  classify    : %.1fx", dtClassCPU / max(dtClassGPU, 0.0001)))
    print(String(format: "  validate    : %.1fx", dtValCPU  / max(dtValGPU,  0.0001)))
    print(String(format: "  isolate     : %.1fx", dtIsoCPU  / max(dtIsoGPU,  0.0001)))
    print(String(format: "  reprocess   : %.1fx", dtRepCPU  / max(dtRepGPU,  0.0001)))
}

func cmdMakeSampleBig(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    try SampleImage.generate(to: url, width: 1024, height: 1024)
    print("sample written: \(url.path) (1024 x 1024)")
}

func cmdFused(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("== XLD Fused ==")
    print("image: \(W) x \(H)  (\(W * H) px)")

    let cpuClass = PerPixelClassifier()
    let cpuVal   = PixelValidator()
    let cpuRep   = ErrorReprocessor()
    let cpuIso   = ErrorIsolator()

    var t = Date()
    let lCPU = cpuClass.classify(img)
    let dtClass = Date().timeIntervalSince(t)
    t = Date()
    let v1CPU = cpuVal.validate(labels: lCPU, width: W, height: H)
    let dtVal1 = Date().timeIntervalSince(t)
    t = Date()
    let _ = cpuIso.isolate(errorMask: v1CPU.errorMask, width: W, height: H)
    let dtIso = Date().timeIntervalSince(t)
    t = Date()
    let repCPU = cpuRep.reprocess(labels: lCPU, errorMask: v1CPU.errorMask, width: W, height: H)
    let dtRep = Date().timeIntervalSince(t)
    t = Date()
    let v2CPU = cpuVal.validate(labels: repCPU.labels, width: W, height: H)
    let dtVal2 = Date().timeIntervalSince(t)
    let cpuTotal = (dtClass + dtVal1 + dtIso + dtRep + dtVal2) * 1000

    print("")
    print("-- CPU total --")
    print(String(format: "  classify    : %.1f ms", dtClass * 1000))
    print(String(format: "  validate #1 : %.1f ms", dtVal1 * 1000))
    print(String(format: "  isolate     : %.1f ms", dtIso * 1000))
    print(String(format: "  reprocess   : %.1f ms", dtRep * 1000))
    print(String(format: "  validate #2 : %.1f ms", dtVal2 * 1000))
    print(String(format: "  TOTAL       : %.1f ms", cpuTotal))
    print(String(format: "  accepted    : %d  errors=%d -> errors=%d",
                 W*H, v1CPU.errorCount, v2CPU.errorCount))

    guard let gpu = MetalPipeline() else { print("gpu unavailable"); return }

    guard let r = gpu.runFullPipeline(image: img) else { print("gpu failed"); return }

    print("")
    print("-- GPU fused (1 cmd buffer, 5 kernels) --")
    print(String(format: "  total pixels: %d", r.totalPixels))
    print(String(format: "  pass1       : accepted=%d  errors=%d",
                 r.pass1Accepted, r.pass1Errors))
    print(String(format: "  error tiles : %d", r.errorTiles))
    print(String(format: "  pass2       : accepted=%d  errors=%d",
                 r.pass2Accepted, r.pass2Errors))
    print(String(format: "  gpu time    : %.2f ms", r.gpuMs))
    print(String(format: "  speedup     : %.1fx", cpuTotal / max(r.gpuMs, 0.001)))

    print("")
    print("-- Agreement (CPU vs GPU) --")
    print(String(format: "  pass1 errors: cpu=%d gpu=%d", v1CPU.errorCount, r.pass1Errors))
    print(String(format: "  pass2 errors: cpu=%d gpu=%d", v2CPU.errorCount, r.pass2Errors))
}

func cmdLayers(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let outDir = url.deletingLastPathComponent().appendingPathComponent("out")
    try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    let t0 = Date()
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("== XLD Layers ==")
    print("image: \(W) x \(H)  (\(W * H) px)")

    guard let gpu = MetalPipeline() else { print("gpu unavailable"); return }

    let tGPU = Date()
    guard let pipe = gpu.runFullPipeline(image: img) else { print("gpu failed"); return }
    let dtGPU = Date().timeIntervalSince(tGPU)

    let labels = gpu.classify(image: img) ?? []
    guard labels.count == W * H else { print("classify failed"); return }

    var flat = [UInt8](repeating: 0, count: W * H)
    for i in 0..<(W * H) where labels[i].classID == 0 { flat[i] = 1 }

    let tCC = Date()
    let cc = ConnectedComponents(minSize: 32)
    let labeling = cc.label(mask: flat, width: W, height: H)
    let dtCC = Date().timeIntervalSince(tCC)

    var kept = 0
    for s in labeling.componentSizes where s >= 32 { kept += 1 }

    print("")
    print("== Connected Components ==")
    print(String(format: "  total comps   : %d", labeling.componentCount))
    print(String(format: "  kept (>=32px) : %d", kept))
    print(String(format: "  cc time       : %.1f ms", dtCC * 1000))

    let tLC = Date()
    let lc = LayerClassifier()
    let assignment = lc.assign(labeling: labeling, width: W, height: H)
    let dtLC = Date().timeIntervalSince(tLC)

    print("")
    print("== Layer Assignment ==")
    let ordered: [LayerID] = [.background, .hairFront, .face, .eyeL, .eyeR, .mouth]
    for layer in ordered {
        if let compID = assignment.layerComponent[layer] {
            let size = assignment.layerPixels[layer] ?? 0
            let bb = labeling.componentBBoxes[compID]
            print(String(format: "  %-12@  comp=%-4d  size=%-6d  bbox=(%4d,%4d)-(%4d,%4d)",
                         layer.rawValue as NSString, compID, size,
                         bb.x0, bb.y0, bb.x1, bb.y1))
        } else {
            print(String(format: "  %-12@  (not found)", layer.rawValue as NSString))
        }
    }
    print(String(format: "  classify time: %.1f ms", dtLC * 1000))

    let visURL = outDir.appendingPathComponent("components.png")
    try ComponentVisualizer.writeVisualization(labeling, width: W, height: H, url: visURL)
    print("")
    print("  visual: \(visURL.path)")

    // per-layer mask 시각화
    var layerIndex: [LayerID: UInt8] = [:]
    var rank: UInt8 = 1
    for layer in ordered {
        if let compID = assignment.layerComponent[layer] {
            layerIndex[layer] = rank
            rank += 1

            var rgba = [Float](repeating: 0, count: W * H * 4)
            for i in 0..<(W * H) {
                if labeling.componentID[i] == Int32(compID) {
                    rgba[i * 4 + 0] = 1.0
                    rgba[i * 4 + 1] = 1.0
                    rgba[i * 4 + 2] = 1.0
                }
                rgba[i * 4 + 3] = 1.0
            }
            let mURL = outDir.appendingPathComponent("layer_\(layer.rawValue).png")
            let rimg = RasterImage(width: W, height: H, rgba: rgba)
            try ImageIO.save(image: rimg, url: mURL)
            print("  wrote \(mURL.lastPathComponent)")
        }
    }

    print("")
    print(String(format: "  gpu pipeline: %.2f ms", dtGPU * 1000))
    print(String(format: "  total time  : %.1f ms", Date().timeIntervalSince(t0) * 1000))
}

func cmdBenchScaling() throws {
    print("== XLD Scaling Bench (fused, 5 kernels) ==")
    print("size      px        cpu(ms)    gpu(ms)    speedup")
    let sizes = [128, 256, 512, 1024, 2048]
    let tmp = URL(fileURLWithPath: "/tmp/xld_bench_sample.png")

    guard let gpu = MetalPipeline() else { print("gpu unavailable"); return }

    let cpuClass = PerPixelClassifier()
    let cpuVal   = PixelValidator()
    let cpuIso   = ErrorIsolator()
    let cpuRep   = ErrorReprocessor()

    for sz in sizes {
        try SampleImage.generate(to: tmp, width: sz, height: sz)
        let img = try ImageIO.load(url: tmp)
        let W = sz, H = sz

        var t = Date()
        let lCPU = cpuClass.classify(img)
        let v1   = cpuVal.validate(labels: lCPU, width: W, height: H)
        _ = cpuIso.isolate(errorMask: v1.errorMask, width: W, height: H)
        let repCPU = cpuRep.reprocess(labels: lCPU, errorMask: v1.errorMask, width: W, height: H)
        _ = cpuVal.validate(labels: repCPU.labels, width: W, height: H)
        let cpuMs = Date().timeIntervalSince(t) * 1000

        t = Date()
        guard let _ = gpu.runFullPipeline(image: img) else {
            print("gpu failed @ \(sz)"); continue
        }
        let gpuMs = Date().timeIntervalSince(t) * 1000

        print(String(format: "%-9d %-9d %-10.1f %-10.2f %.1fx",
                     sz, sz * sz, cpuMs, gpuMs, cpuMs / max(gpuMs, 0.001)))
    }
}

enum SampleImage {
    static func generate(to url: URL, width: Int, height: Int) throws {
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil,
                                  width: width, height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: width * 4,
                                  space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ImageIOError.saveFailed }

        ctx.setFillColor(CGColor(red: 0.95, green: 0.95, blue: 0.95, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))

        ctx.setFillColor(CGColor(red: 1.00, green: 0.86, blue: 0.76, alpha: 1))
        ctx.fillEllipse(in: CGRect(x: 60, y: 40, width: 136, height: 160))

        ctx.setFillColor(CGColor(red: 0.25, green: 0.18, blue: 0.12, alpha: 1))
        ctx.fillEllipse(in: CGRect(x: 50, y: 170, width: 156, height: 70))

        ctx.setFillColor(CGColor(red: 0.10, green: 0.10, blue: 0.15, alpha: 1))
        ctx.fillEllipse(in: CGRect(x:  95, y: 130, width: 20, height: 12))
        ctx.fillEllipse(in: CGRect(x: 145, y: 130, width: 20, height: 12))

        ctx.setStrokeColor(CGColor(red: 0.65, green: 0.20, blue: 0.20, alpha: 1))
        ctx.setLineWidth(3)
        ctx.move(to: CGPoint(x: 110, y: 90))
        ctx.addLine(to: CGPoint(x: 150, y: 90))
        ctx.strokePath()

        var rng = SystemRandomNumberGenerator()
        for _ in 0..<900 {
            let x = Int.random(in: 0..<width,  using: &rng)
            let y = Int.random(in: 0..<height, using: &rng)
            ctx.setFillColor(CGColor(red: Double.random(in: 0.3...0.9, using: &rng),
                                     green: Double.random(in: 0.3...0.9, using: &rng),
                                     blue: Double.random(in: 0.3...0.9, using: &rng),
                                     alpha: 0.15))
            ctx.fill(CGRect(x: x, y: y, width: 1, height: 1))
        }

        guard let img = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL,
                                                          "public.png" as CFString, 1, nil)
        else { throw ImageIOError.saveFailed }

        CGImageDestinationAddImage(dest, img, nil)
        if !CGImageDestinationFinalize(dest) { throw ImageIOError.saveFailed }
    }
}

var mode = "auto"
var positional: [String] = []
for a in args {
    if a == "--gpu" { mode = "gpu" }
    else if a == "--cpu" { mode = "cpu" }
    else if a == "--auto" { mode = "auto" }
    else { positional.append(a) }
}

if mode == "auto" {
    mode = (MetalRuntime.shared != nil) ? "gpu" : "cpu"
}

do {
    switch positional.first {
    case "version", "--version", "-v":
        cmdVersion()
    case "pipeline":
        cmdPipeline()
    case "make-sample":
        guard positional.count >= 2 else { usage(); exit(1) }
        try cmdMakeSample(positional[1])
    case "run":
        guard positional.count >= 2 else { usage(); exit(1) }
        try cmdRun(positional[1], mode: mode)
    case "bench":
        guard positional.count >= 2 else { usage(); exit(1) }
        try cmdBench(positional[1])
    case "fused":
        guard positional.count >= 2 else { usage(); exit(1) }
        try cmdFused(positional[1])
    case "make-sample-big":
        guard positional.count >= 2 else { usage(); exit(1) }
        try cmdMakeSampleBig(positional[1])
    case "layers":
        guard positional.count >= 2 else { usage(); exit(1) }
        try cmdLayers(positional[1])
    case "bench-scaling":
        try cmdBenchScaling()
    case "bench-gpu":
        guard positional.count >= 2 else { usage(); exit(1) }
        try cmdBenchGPU(positional[1])
    default:
        usage()
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
