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

func cmdBenchSim(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("== XLD Bench-Sim ==")
    print("image: \(W) x \(H)  (\(W * H) px)")

    guard let gpu = MetalPipeline() else { print("gpu unavailable"); return }
    guard let labels = gpu.classify(image: img), labels.count == W * H else {
        print("classify failed"); return
    }

    var flat = [UInt8](repeating: 0, count: W * H)
    var edge = [UInt8](repeating: 0, count: W * H)
    for i in 0..<(W * H) {
        if labels[i].classID == 0 { flat[i] = 1 }
        if labels[i].classID == 1 { edge[i] = 1 }
    }
    let flatLabeling = ConnectedComponents(minSize: 32).label(mask: flat, width: W, height: H)
    let edgeLabeling = ConnectedComponents(minSize: 8).label(mask: edge, width: W, height: H)
    let assignment = LayerClassifier().assign(flatLabeling: flatLabeling,
                                              edgeLabeling: edgeLabeling,
                                              width: W, height: H)

    var layerMasks: [LayerID: [UInt8]] = [:]
    for (layer, compID) in assignment.layerComponent {
        let src = assignment.layerSource[layer] ?? .flatComponent
        let labeling = (src == .edgeComponent) ? edgeLabeling : flatLabeling
        var mask = [UInt8](repeating: 0, count: W * H)
        for i in 0..<(W * H) where labeling.componentID[i] == Int32(compID) {
            mask[i] = 1
        }
        layerMasks[layer] = mask
    }

    let occ = OcclusionGraphBuilder().build(layerMasks: layerMasks, width: W, height: H)
    let seq = DefaultMotionSequences.vtuber()

    // CPU
    let cpuSim = MotionSimulator()
    var t = Date()
    let cpuResult = cpuSim.simulate(sequence: seq,
                                    layerMasks: layerMasks,
                                    layerZ: occ.layerZ,
                                    originalTopRank: occ.pixelOwnerRank,
                                    orderedLayers: occ.orderedLayers,
                                    width: W, height: H)
    let cpuMs = Date().timeIntervalSince(t) * 1000

    // GPU
    guard let gpuSim = MetalMotionSimulator() else {
        print("gpu sim unavailable"); return
    }
    t = Date()
    guard let gpuResult = gpuSim.simulate(sequence: seq,
                                          layerMasks: layerMasks,
                                          layerZ: occ.layerZ,
                                          originalTopRank: occ.pixelOwnerRank,
                                          orderedLayers: occ.orderedLayers,
                                          width: W, height: H) else {
        print("gpu sim failed"); return
    }
    let gpuMs = Date().timeIntervalSince(t) * 1000

    print("")
    print("-- Timing --")
    print(String(format: "  cpu sim   : %.1f ms", cpuMs))
    print(String(format: "  gpu sim   : %.1f ms", gpuMs))
    print(String(format: "  speedup   : %.1fx", cpuMs / max(gpuMs, 0.001)))

    print("")
    print("-- Lifetime Agreement --")
    print(String(format: "  lifetime px cpu=%d gpu=%d",
                 cpuResult.lifetimeCount, gpuResult.lifetimeCount))

    var mismatch = 0
    for i in 0..<(W * H) {
        if cpuResult.lifetimeDisocclusion[i] != gpuResult.lifetimeDisocclusion[i] { mismatch += 1 }
    }
    print(String(format: "  lifetime mismatches: %d", mismatch))

    print("")
    print("-- Pose count --")
    print(String(format: "  cpu poses: %d", cpuResult.poseResults.count))
    print(String(format: "  gpu poses: %d", gpuResult.poseResults.count))

    if !cpuResult.poseResults.isEmpty, !gpuResult.poseResults.isEmpty {
        print("")
        print("  pose              cpu px    gpu px    diff")
        let n = min(cpuResult.poseResults.count, gpuResult.poseResults.count)
        var maxDiff = 0
        for i in 0..<n {
            let c = cpuResult.poseResults[i]
            let g = gpuResult.poseResults[i]
            let d = abs(c.disocclusionCount - g.disocclusionCount)
            if d > maxDiff { maxDiff = d }
            print(String(format: "  %-16@  %6d    %6d    %d",
                         c.name as NSString, c.disocclusionCount, g.disocclusionCount, d))
        }
        print(String(format: "  max diff: %d", maxDiff))
    }
}

func cmdDiagnose(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let outDir = url.deletingLastPathComponent().appendingPathComponent(
        url.deletingPathExtension().lastPathComponent + "_diag")
    try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    let t0 = Date()
    print("====================================================")
    print(" XLD L2D — DIAGNOSE v3")
    print("====================================================")

    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    let total = W * H
    print("file:  \(url.path)")
    print("size:  \(W) x \(H)  (\(W * H) px)")

    // --- [1] GPU classify + edge ---
    guard let gpu = MetalPipeline() else { print("metal unavailable"); return }
    let tPipe = Date()
    guard let pipe = gpu.runFullPipeline(image: img) else {
        print("pipeline failed"); return
    }
    let dtPipe = Date().timeIntervalSince(tPipe)
    print("")
    print("== [1] classify + validate ==")
    print(String(format: "  pass1 errs=%d (%.2f%%)  pass2 errs=%d (%.2f%%)  time=%.1f ms",
                 pipe.pass1Errors, Float(pipe.pass1Errors) / Float(total) * 100,
                 pipe.pass2Errors, Float(pipe.pass2Errors) / Float(total) * 100,
                 dtPipe * 1000))

    guard let labels = gpu.classify(image: img), labels.count == total else {
        print("classify failed"); return
    }
    var edge = [UInt8](repeating: 0, count: total)
    var region = [UInt8](repeating: 0, count: total)
    for i in 0..<total {
        let c = labels[i].classID
        if c == 1 { edge[i] = 1 } else { region[i] = 1 }
    }

    // --- [2] background removal ---
    print("")
    let tBG = Date()
    let bgRes = BackgroundRemover(tolerance: 0.18, diagonal: true).remove(image: img)
    let dtBG = Date().timeIntervalSince(tBG)
    print("== [2] background removal ==")
    print(String(format: "  bg=%.2f%%  char=%.2f%%  time=%.1f ms",
                 Float(bgRes.backgroundCount) / Float(total) * 100,
                 Float(bgRes.characterCount) / Float(total) * 100,
                 dtBG * 1000))

    var bgRGBA = [Float](repeating: 0, count: total * 4)
    for i in 0..<total {
        if bgRes.backgroundMask[i] == 1 {
            bgRGBA[i*4] = 0.1; bgRGBA[i*4+1] = 0.1; bgRGBA[i*4+2] = 0.15
        } else {
            bgRGBA[i*4] = img.rgba[i*4]
            bgRGBA[i*4+1] = img.rgba[i*4+1]
            bgRGBA[i*4+2] = img.rgba[i*4+2]
        }
        bgRGBA[i*4+3] = 1.0
    }
    let bgURL = outDir.appendingPathComponent("background.png")
    try ImageIO.save(image: RasterImage(width: W, height: H, rgba: bgRGBA), url: bgURL)

    // --- [3] hair detection ---
    print("")
    let tHair = Date()
    let hair = HairDetector(topFraction: 0.42).detect(image: img,
                                                       characterMask: bgRes.characterMask)
    let dtHair = Date().timeIntervalSince(tHair)
    print("== [3] hair detection (top 42%, blue) ==")
    if let cb = hair.charBBox {
        print(String(format: "  char bbox : (%4d,%4d)-(%4d,%4d)  %dx%d",
                     cb.0, cb.1, cb.2, cb.3, cb.2 - cb.0 + 1, cb.3 - cb.1 + 1))
    }
    print(String(format: "  hair px: %d (%.2f%%)  time=%.1f ms",
                 hair.hairCount, Float(hair.hairCount) / Float(total) * 100,
                 dtHair * 1000))
    if let hb = hair.bbox {
        print(String(format: "  hair bbox: (%4d,%4d)-(%4d,%4d)  %dx%d",
                     hb.x0, hb.y0, hb.x1, hb.y1,
                     hb.x1 - hb.x0 + 1, hb.y1 - hb.y0 + 1))
    }

    var hairRGBA = [Float](repeating: 0, count: total * 4)
    for i in 0..<total {
        if hair.hairMask[i] == 1 {
            hairRGBA[i*4] = 0.2; hairRGBA[i*4+1] = 0.6; hairRGBA[i*4+2] = 1.0
        } else if bgRes.characterMask[i] == 1 {
            hairRGBA[i*4] = 0.3; hairRGBA[i*4+1] = 0.3; hairRGBA[i*4+2] = 0.3
        }
        hairRGBA[i*4+3] = 1.0
    }
    let hairURL = outDir.appendingPathComponent("hair.png")
    try ImageIO.save(image: RasterImage(width: W, height: H, rgba: hairRGBA), url: hairURL)

    // --- [4] face finder v4 (hair-enclosed) ---
    print("")
    let tF4 = Date()
    let faceRes = FaceFinderV4(hairDilation: 24, minFacePx: 2000,
                               maxFacePct: 0.06, topFraction: 0.55)
        .find(image: img, characterMask: bgRes.characterMask,
              hairMask: hair.hairMask, edgeMask: edge)
    let dtF4 = Date().timeIntervalSince(tF4)
    print("== [4] face finder v4 (hair-enclosed region) ==")
    print(String(format: "  hair dilated px: %d", faceRes.hairDilatedCount))
    print(String(format: "  enclosed px: %d  time=%.1f ms",
                 faceRes.enclosedCount, dtF4 * 1000))
    if let f = faceRes.faceBox {
        print(String(format: "  face bbox: (%4d,%4d)-(%4d,%4d)  %dx%d  score=%.1f",
                     f.x0, f.y0, f.x1, f.y1,
                     f.x1 - f.x0 + 1, f.y1 - f.y0 + 1, faceRes.score))
    } else {
        print("  *** face not found ***")
    }

    // face_v3 visualization
    var faceRGBA = [Float](repeating: 0, count: total * 4)
    for i in 0..<total {
        if faceRes.faceMask[i] == 1 {
            faceRGBA[i*4] = 1.0; faceRGBA[i*4+1] = 0.85; faceRGBA[i*4+2] = 0.65
        } else if hair.hairMask[i] == 1 {
            faceRGBA[i*4] = 0.20; faceRGBA[i*4+1] = 0.55; faceRGBA[i*4+2] = 1.0
        } else if bgRes.characterMask[i] == 1 {
            faceRGBA[i*4] = 0.30; faceRGBA[i*4+1] = 0.30; faceRGBA[i*4+2] = 0.30
        } else {
            faceRGBA[i*4] = 0.05; faceRGBA[i*4+1] = 0.05; faceRGBA[i*4+2] = 0.05
        }
        faceRGBA[i*4+3] = 1.0
    }
    let faceV3URL = outDir.appendingPathComponent("face_v3.png")
    try ImageIO.save(image: RasterImage(width: W, height: H, rgba: faceRGBA), url: faceV3URL)

    // face box + border
    if let f = faceRes.faceBox {
        var ov = img.rgba
        for y in 0..<H {
            for x in 0..<W {
                let inX = x >= f.x0 && x <= f.x1
                let inY = y >= f.y0 && y <= f.y1
                if inX && inY {
                    let i = (y * W + x) * 4
                    let isBorder = (x == f.x0 || x == f.x1 || y == f.y0 || y == f.y1)
                    if isBorder {
                        ov[i] = 1.0; ov[i+1] = 0.0; ov[i+2] = 1.0
                    } else {
                        ov[i] = ov[i] * 0.6 + 0.4
                        ov[i+1] = ov[i+1] * 0.6
                        ov[i+2] = ov[i+2] * 0.6
                    }
                }
            }
        }
        let fbURL = outDir.appendingPathComponent("facebox.png")
        try ImageIO.save(image: RasterImage(width: W, height: H, rgba: ov), url: fbURL)
    }

    print("")
    print("== artifacts ==")
    print("  \(bgURL.path)")
    print("  \(hairURL.path)")
    print("  \(faceV3URL.path)")
    if let f = faceRes.faceBox {
        _ = f
        print("  \(outDir.appendingPathComponent("facebox.png").path)")
    }
    print("")
    print(String(format: "  total: %.1f ms", Date().timeIntervalSince(t0) * 1000))
    print("====================================================")
}

func cmdPipelineFull(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let outDir = url.deletingLastPathComponent().appendingPathComponent("out")
    try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    let t0 = Date()
    print("====================================================")
    print(" XLD L2D — FULL PIPELINE")
    print("====================================================")
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("image: \(url.lastPathComponent)  \(W) x \(H)  (\(W * H) px)")
    print("")

    // stage 1: GPU classify + validate + reprocess
    guard let gpu = MetalPipeline() else { print("gpu unavailable"); return }
    let tPipe = Date()
    guard let pipeRes = gpu.runFullPipeline(image: img) else { print("gpu failed"); return }
    let dtPipe = Date().timeIntervalSince(tPipe)
    print(String(format: "[1/8] GPU pipeline        %.2f ms   pass1 err=%d  pass2 err=%d  tiles=%d",
                 dtPipe * 1000, pipeRes.pass1Errors, pipeRes.pass2Errors, pipeRes.errorTiles))

    // stage 2: layers
    let tLayer = Date()
    guard let labels = gpu.classify(image: img), labels.count == W * H else {
        print("classify failed"); return
    }
    var flat = [UInt8](repeating: 0, count: W * H)
    var edge = [UInt8](repeating: 0, count: W * H)
    for i in 0..<(W * H) {
        if labels[i].classID == 0 { flat[i] = 1 }
        if labels[i].classID == 1 { edge[i] = 1 }
    }
    let flatLabeling = ConnectedComponents(minSize: 32).label(mask: flat, width: W, height: H)
    let edgeLabeling = ConnectedComponents(minSize: 8).label(mask: edge, width: W, height: H)
    let assignment = LayerClassifier().assign(flatLabeling: flatLabeling,
                                              edgeLabeling: edgeLabeling,
                                              width: W, height: H)
    var layerMasks: [LayerID: [UInt8]] = [:]
    for (layer, compID) in assignment.layerComponent {
        let src = assignment.layerSource[layer] ?? .flatComponent
        let labeling = (src == .edgeComponent) ? edgeLabeling : flatLabeling
        var mask = [UInt8](repeating: 0, count: W * H)
        for i in 0..<(W * H) where labeling.componentID[i] == Int32(compID) {
            mask[i] = 1
        }
        layerMasks[layer] = mask
    }
    let dtLayer = Date().timeIntervalSince(tLayer)
    print(String(format: "[2/8] Layer separation     %.2f ms   %d layers", dtLayer * 1000, layerMasks.count))

    // stage 3: occlusion
    let tOcc = Date()
    let occGraph = OcclusionGraphBuilder().build(layerMasks: layerMasks, width: W, height: H)
    let dtOcc = Date().timeIntervalSince(tOcc)
    print(String(format: "[3/8] Occlusion graph      %.2f ms   order=%@",
                 dtOcc * 1000,
                 occGraph.orderedLayers.map { $0.rawValue }.joined(separator: " < ") as NSString))

    // stage 4: envelope
    let tEnv = Date()
    let envelope = MovementEnvelopeBuilder(budget: 20).build(layerMasks: layerMasks,
                                                             width: W, height: H)
    let dtEnv = Date().timeIntervalSince(tEnv)
    let totalBudget = envelope.layers.values.reduce(0) { $0 + $1.disocclusionBudget }
    print(String(format: "[4/8] Movement envelope    %.2f ms   total disocc budget=%d px",
                 dtEnv * 1000, totalBudget))

    // stage 5: lifetime simulation
    let tSim = Date()
    let seq = DefaultMotionSequences.vtuber()
    let sim = MotionSimulator()
    let lifetime = sim.simulate(sequence: seq,
                                layerMasks: layerMasks,
                                layerZ: occGraph.layerZ,
                                originalTopRank: occGraph.pixelOwnerRank,
                                orderedLayers: occGraph.orderedLayers,
                                width: W, height: H)
    let dtSim = Date().timeIntervalSince(tSim)
    let lifetimePct = Float(lifetime.lifetimeCount) / Float(W * H) * 100.0
    print(String(format: "[5/8] Lifetime simulation  %.2f ms   %d poses, disocc=%d px (%.2f%%)",
                 dtSim * 1000, lifetime.poseResults.count, lifetime.lifetimeCount, lifetimePct))

    // stage 6: minimal inpainting
    let tInp = Date()
    let engine = MinimalInpaintingEngine()
    let req = InpaintRequest(width: W, height: H,
                             disocclusionMask: lifetime.lifetimeDisocclusion,
                             contextRGBA: img.rgba,
                             tileSize: 16,
                             maxPixels: W * H)
    guard let out = engine.inpaint(request: req) else { print("inpaint failed"); return }
    let dtInp = Date().timeIntervalSince(tInp)
    print(String(format: "[6/8] Minimal inpainting   %.2f ms   tiles=%d  px=%d",
                 dtInp * 1000, out.stats.tilesInpainted, out.stats.pixelsTouched))

    // stage 7: write artifacts
    let tWrite = Date()
    let inpDir = outDir.appendingPathComponent("inpaint")
    try? FileManager.default.createDirectory(at: inpDir, withIntermediateDirectories: true)
    let restoredURL = inpDir.appendingPathComponent("restored.png")
    try ImageIO.save(image: RasterImage(width: W, height: H, rgba: out.rgba), url: restoredURL)

    let ltDir = outDir.appendingPathComponent("lifetime")
    try? FileManager.default.createDirectory(at: ltDir, withIntermediateDirectories: true)
    try LifetimeVisualizer.writeLifetime(lifetime, url: ltDir.appendingPathComponent("lifetime_disocclusion.png"))
    try LifetimeVisualizer.writeLifetimeOverlay(lifetime, original: img,
                                                url: ltDir.appendingPathComponent("lifetime_overlay.png"))

    var compIDs: [String: Int] = [:]
    var sizes:   [String: Int] = [:]
    var srcMap:  [String: String] = [:]
    var layerMasksStr: [String: [UInt8]] = [:]
    for (layer, compID) in assignment.layerComponent {
        compIDs[layer.rawValue] = compID
        sizes[layer.rawValue] = assignment.layerPixels[layer] ?? 0
        srcMap[layer.rawValue] = assignment.layerSource[layer]?.rawValue ?? "?"
    }
    for (layer, mask) in layerMasks { layerMasksStr[layer.rawValue] = mask }

    let doc = XLDModelDocument(
        version: XLDKit.version,
        imagePath: url.lastPathComponent,
        imageWidth: W,
        imageHeight: H,
        layerComponentID: compIDs,
        layerPixelCount: sizes,
        layerSource: srcMap,
        layerMasks: layerMasksStr,
        occlusionGraph: occGraph,
        movementEnvelope: envelope,
        lifetimeDisocclusion: lifetime.lifetimeDisocclusion,
        lifetimeSequenceName: lifetime.sequenceName,
        lifetimePoseCount: lifetime.poseResults.count
    )
    let modelDir = outDir.appendingPathComponent("model.xldmodel")
    try XLDModelIO.write(doc, to: modelDir)
    let dtWrite = Date().timeIntervalSince(tWrite)
    print(String(format: "[7/8] Artifact writing     %.2f ms   %@",
                 dtWrite * 1000, modelDir.lastPathComponent as NSString))

    // stage 8: report
    let tTotal = Date().timeIntervalSince(t0)
    print(String(format: "[8/8] Total                %.2f ms", tTotal * 1000))
    print("")
    print("====================================================")
    print(" RESULT")
    print("====================================================")
    print(String(format: "  original preserved : %.2f%%", 100.0 - lifetimePct))
    print(String(format: "  needs inpaint      : %.2f%%", lifetimePct))
    print(String(format: "  layer count        : %d", layerMasks.count))
    print(String(format: "  disocclusion px    : %d", lifetime.lifetimeCount))
    print(String(format: "  inpaint tiles      : %d", out.stats.tilesInpainted))
    print("")
    print("  artifacts:")
    print("    \(modelDir.path)/")
    print("    \(restoredURL.path)")
    print("    \(ltDir.path)/lifetime_overlay.png")
    print("====================================================")
}

func cmdInpaint(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let outDir = url.deletingLastPathComponent().appendingPathComponent("out")
    let inpDir = outDir.appendingPathComponent("inpaint")
    try? FileManager.default.createDirectory(at: inpDir, withIntermediateDirectories: true)

    let t0 = Date()
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("== XLD Minimal Inpainting ==")
    print("image: \(url.lastPathComponent)  \(W) x \(H)")

    // rebuild layers + occlusion
    guard let gpu = MetalPipeline() else { print("gpu unavailable"); return }
    guard let labels = gpu.classify(image: img), labels.count == W * H else {
        print("classify failed"); return
    }
    var flat = [UInt8](repeating: 0, count: W * H)
    var edge = [UInt8](repeating: 0, count: W * H)
    for i in 0..<(W * H) {
        if labels[i].classID == 0 { flat[i] = 1 }
        if labels[i].classID == 1 { edge[i] = 1 }
    }
    let flatLabeling = ConnectedComponents(minSize: 32).label(mask: flat, width: W, height: H)
    let edgeLabeling = ConnectedComponents(minSize: 8).label(mask: edge, width: W, height: H)
    let lc = LayerClassifier()
    let assignment = lc.assign(flatLabeling: flatLabeling,
                               edgeLabeling: edgeLabeling,
                               width: W, height: H)

    var layerMasks: [LayerID: [UInt8]] = [:]
    for (layer, compID) in assignment.layerComponent {
        let src = assignment.layerSource[layer] ?? .flatComponent
        let labeling = (src == .edgeComponent) ? edgeLabeling : flatLabeling
        var mask = [UInt8](repeating: 0, count: W * H)
        for i in 0..<(W * H) where labeling.componentID[i] == Int32(compID) {
            mask[i] = 1
        }
        layerMasks[layer] = mask
    }

    let occGraph = OcclusionGraphBuilder().build(layerMasks: layerMasks, width: W, height: H)

    let seq = DefaultMotionSequences.vtuber()
    let lifetime = MotionSimulator().simulate(sequence: seq,
                                              layerMasks: layerMasks,
                                              layerZ: occGraph.layerZ,
                                              originalTopRank: occGraph.pixelOwnerRank,
                                              orderedLayers: occGraph.orderedLayers,
                                              width: W, height: H)

    print("")
    print("== Lifetime Mask ==")
    print(String(format: "  disocclusion px: %d  (%.2f%%)",
                 lifetime.lifetimeCount,
                 Float(lifetime.lifetimeCount) / Float(W * H) * 100))

    // run inpaint engine (stubs)
    let engine = MinimalInpaintingEngine()
    let req = InpaintRequest(width: W, height: H,
                             disocclusionMask: lifetime.lifetimeDisocclusion,
                             contextRGBA: img.rgba,
                             tileSize: 16,
                             maxPixels: W * H)
    guard let out = engine.inpaint(request: req) else {
        print("inpaint failed"); return
    }

    print("")
    print("== Inpaint Engine ==")
    print("  structure     : \(engine.structure.name) (\(engine.structure.version))")
    print("  appearance    : \(engine.appearance.name) (\(engine.appearance.version))")
    print("  tiles req     : \(out.stats.tilesRequested)")
    print("  tiles done    : \(out.stats.tilesInpainted)")
    print("  pixels touched: \(out.stats.pixelsTouched)")
    print("  cache hits    : \(out.stats.cacheHits)")
    print("  cache misses  : \(out.stats.cacheMisses)")
    print(String(format: "  inpaint time  : %.2f ms", out.stats.gpuMs))

    // restore + save
    let restored = RasterImage(width: W, height: H, rgba: out.rgba)
    let restoredURL = inpDir.appendingPathComponent("restored.png")
    try ImageIO.save(image: restored, url: restoredURL)

    // comparison: 50/50 blend of original and restored
    var blend = img.rgba
    for i in 0..<(W * H * 4) {
        blend[i] = img.rgba[i] * 0.5 + out.rgba[i] * 0.5
    }
    let blendURL = inpDir.appendingPathComponent("blend.png")
    try ImageIO.save(image: RasterImage(width: W, height: H, rgba: blend), url: blendURL)

    print("")
    print("== Artifacts ==")
    print("  \(restoredURL.path)")
    print("  \(blendURL.path)")

    print("")
    print(String(format: "  total time: %.1f ms", Date().timeIntervalSince(t0) * 1000))
}

func cmdSimulate(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let outDir = url.deletingLastPathComponent().appendingPathComponent("out")
    let simDir = outDir.appendingPathComponent("lifetime")
    try? FileManager.default.createDirectory(at: simDir, withIntermediateDirectories: true)

    let t0 = Date()
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("== XLD Lifetime Simulation ==")
    print("image: \(url.lastPathComponent)  \(W) x \(H)  (\(W * H) px)")

    // --- rebuild layers ---
    guard let gpu = MetalPipeline() else { print("gpu unavailable"); return }
    guard let labels = gpu.classify(image: img), labels.count == W * H else {
        print("classify failed"); return
    }

    var flat = [UInt8](repeating: 0, count: W * H)
    var edge = [UInt8](repeating: 0, count: W * H)
    for i in 0..<(W * H) {
        if labels[i].classID == 0 { flat[i] = 1 }
        if labels[i].classID == 1 { edge[i] = 1 }
    }

    let flatLabeling = ConnectedComponents(minSize: 32).label(mask: flat, width: W, height: H)
    let edgeLabeling = ConnectedComponents(minSize: 8).label(mask: edge, width: W, height: H)

    let lc = LayerClassifier()
    let assignment = lc.assign(flatLabeling: flatLabeling,
                               edgeLabeling: edgeLabeling,
                               width: W, height: H)

    var layerMasks: [LayerID: [UInt8]] = [:]
    for (layer, compID) in assignment.layerComponent {
        let src = assignment.layerSource[layer] ?? .flatComponent
        let labeling = (src == .edgeComponent) ? edgeLabeling : flatLabeling
        var mask = [UInt8](repeating: 0, count: W * H)
        for i in 0..<(W * H) where labeling.componentID[i] == Int32(compID) {
            mask[i] = 1
        }
        layerMasks[layer] = mask
    }

    // --- build original occlusion graph ---
    let occBuilder = OcclusionGraphBuilder()
    let occGraph = occBuilder.build(layerMasks: layerMasks, width: W, height: H)

    print("")
    print("== Layers Present ==")
    let orderedLayers: [LayerID] = [.background, .hairFront, .face, .eyeL, .eyeR, .mouth]
    for l in orderedLayers where layerMasks[l] != nil {
        print("  \(l.rawValue)")
    }

    // --- run simulation ---
    let seq = DefaultMotionSequences.vtuber()
    let sim = MotionSimulator()

    let tSim = Date()
    let result = sim.simulate(sequence: seq,
                              layerMasks: layerMasks,
                              layerZ: occGraph.layerZ,
                              originalTopRank: occGraph.pixelOwnerRank,
                              orderedLayers: occGraph.orderedLayers,
                              width: W, height: H)
    let dtSim = Date().timeIntervalSince(tSim)

    print("")
    print("== Sequence: \(result.sequenceName) ==")
    print("  poses: \(result.poseResults.count)")

    print("")
    print("  pose              disocc px   topChanged  unmasked")
    for p in result.poseResults {
        print(String(format: "  %-16@  %8d   %9d  %8d",
                     p.name as NSString, p.disocclusionCount,
                     p.topChangedCount, p.unmaskedCount))
    }

    let lifetimePct = Float(result.lifetimeCount) / Float(result.totalPixels) * 100.0
    let keptPct = 100.0 - lifetimePct

    print("")
    print("== Lifetime Disocclusion ==")
    print(String(format: "  total pixels       : %d", result.totalPixels))
    print(String(format: "  lifetime disocc px : %d", result.lifetimeCount))
    print(String(format: "  needs inpaint      : %.2f%%", lifetimePct))
    print(String(format: "  original preserved : %.2f%%", keptPct))

    // --- write outputs ---
    let ltURL = simDir.appendingPathComponent("lifetime_disocclusion.png")
    let ovURL = simDir.appendingPathComponent("lifetime_overlay.png")
    try LifetimeVisualizer.writeLifetime(result, url: ltURL)
    try LifetimeVisualizer.writeLifetimeOverlay(result, original: img, url: ovURL)

    for p in result.poseResults {
        let purl = simDir.appendingPathComponent("pose_\(p.name).png")
        try LifetimeVisualizer.writePose(p, width: W, height: H, url: purl)
    }

    print("")
    print("== Artifacts ==")
    print("  \(ltURL.path)")
    print("  \(ovURL.path)")
    print("  \(result.poseResults.count) pose images")

    print("")
    print(String(format: "  sim time  : %.1f ms", dtSim * 1000))
    print(String(format: "  total time: %.1f ms", Date().timeIntervalSince(t0) * 1000))
}

func cmdMove(_ path: String, layer layerName: String, dx: Int, dy: Int) throws {
    guard let targetLayer = LayerID(rawValue: layerName) else {
        print("unknown layer: \(layerName)")
        print("valid: \(LayerID.allCases.map { $0.rawValue }.joined(separator: ", "))")
        return
    }

    let url = URL(fileURLWithPath: path)
    let outDir = url.deletingLastPathComponent().appendingPathComponent("out")
    let moveDir = outDir.appendingPathComponent("move_\(layerName)_\(dx)_\(dy)")
    try? FileManager.default.createDirectory(at: moveDir, withIntermediateDirectories: true)

    let t0 = Date()
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("== XLD Move ==")
    print("image:  \(url.lastPathComponent)  \(W) x \(H)")
    print("layer:  \(layerName)")
    print("delta:  (\(dx), \(dy))")

    // build layers (reuse pipeline)
    guard let gpu = MetalPipeline() else { print("gpu unavailable"); return }
    guard let labels = gpu.classify(image: img), labels.count == W * H else {
        print("classify failed"); return
    }

    var flat = [UInt8](repeating: 0, count: W * H)
    var edge = [UInt8](repeating: 0, count: W * H)
    for i in 0..<(W * H) {
        if labels[i].classID == 0 { flat[i] = 1 }
        if labels[i].classID == 1 { edge[i] = 1 }
    }

    let cc = ConnectedComponents(minSize: 32)
    let flatLabeling = cc.label(mask: flat, width: W, height: H)
    let edgeCC = ConnectedComponents(minSize: 8)
    let edgeLabeling = edgeCC.label(mask: edge, width: W, height: H)

    let lc = LayerClassifier()
    let assignment = lc.assign(flatLabeling: flatLabeling,
                               edgeLabeling: edgeLabeling,
                               width: W, height: H)

    guard let compID = assignment.layerComponent[targetLayer] else {
        print("layer \(layerName) not found in this image"); return
    }
    let src = assignment.layerSource[targetLayer] ?? .flatComponent
    let srcLabeling = (src == .edgeComponent) ? edgeLabeling : flatLabeling

    var targetMask = [UInt8](repeating: 0, count: W * H)
    for i in 0..<(W * H) where srcLabeling.componentID[i] == Int32(compID) {
        targetMask[i] = 1
    }

    print("")
    print("== Original Layer ==")
    print(String(format: "  component id : %d", compID))
    print(String(format: "  pixel count  : %d", assignment.layerPixels[targetLayer] ?? 0))

    let mover = LayerMover()
    let result = mover.move(layer: targetLayer, mask: targetMask,
                            width: W, height: H, dx: dx, dy: dy)

    print("")
    print("== Move Result ==")
    print(String(format: "  original px      : %d", result.originalCount))
    print(String(format: "  moved px         : %d", result.movedCount))
    print(String(format: "  clipped (OOB)    : %d", result.clippedCount))
    print(String(format: "  disocclusion px  : %d  (새로 드러남, inpaint 필요)",
                 result.disocclusionCount))
    print(String(format: "  new coverage px  : %d  (새로 덮음)", result.newCoverageCount))

    if result.movedCount > 0 {
        let disRatio = Float(result.disocclusionCount) / Float(result.movedCount) * 100.0
        print(String(format: "  disocclusion / moved : %.1f%%", disRatio))
    }

    let disURL = moveDir.appendingPathComponent("disocclusion.png")
    let ovlURL = moveDir.appendingPathComponent("overlay.png")
    try MoveVisualizer.writeDisocclusion(result, url: disURL)
    try MoveVisualizer.writeOverlay(result, original: img, url: ovlURL)

    print("")
    print("== Artifacts ==")
    print("  \(disURL.path)")
    print("  \(ovlURL.path)")

    print("")
    print(String(format: "  total time: %.1f ms", Date().timeIntervalSince(t0) * 1000))
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
    guard gpu.runFullPipeline(image: img) != nil else { print("gpu failed"); return }
    let dtGPU = Date().timeIntervalSince(tGPU)

    guard let labels = gpu.classify(image: img), labels.count == W * H else {
        print("classify failed"); return
    }

    var flat = [UInt8](repeating: 0, count: W * H)
    var edge = [UInt8](repeating: 0, count: W * H)
    for i in 0..<(W * H) {
        if labels[i].classID == 0 { flat[i] = 1 }
        if labels[i].classID == 1 { edge[i] = 1 }
    }

    let tCC = Date()
    let cc = ConnectedComponents(minSize: 32)
    let flatLabeling = cc.label(mask: flat, width: W, height: H)
    let edgeCC = ConnectedComponents(minSize: 8)
    let edgeLabeling = edgeCC.label(mask: edge, width: W, height: H)
    let dtCC = Date().timeIntervalSince(tCC)

    print("")
    print("== Connected Components ==")
    print(String(format: "  flat comps    : %d", flatLabeling.componentCount))
    print(String(format: "  edge comps    : %d", edgeLabeling.componentCount))
    print(String(format: "  cc time       : %.1f ms", dtCC * 1000))

    let tLC = Date()
    let lc = LayerClassifier()
    let assignment = lc.assign(flatLabeling: flatLabeling,
                               edgeLabeling: edgeLabeling,
                               width: W, height: H)
    _ = Date().timeIntervalSince(tLC)

    print("")
    print("== Layer Assignment ==")
    let ordered: [LayerID] = [.background, .hairFront, .face, .eyeL, .eyeR, .mouth]
    for layer in ordered {
        if let compID = assignment.layerComponent[layer] {
            let size = assignment.layerPixels[layer] ?? 0
            let src = assignment.layerSource[layer]?.rawValue ?? "?"
            let labeling = (src == "edgeComponent") ? edgeLabeling : flatLabeling
            let bb = labeling.componentBBoxes[compID]
            print(String(format: "  %-12@  comp=%-4d  size=%-6d  src=%-15@  bbox=(%4d,%4d)-(%4d,%4d)",
                         layer.rawValue as NSString, compID, size, src as NSString,
                         bb.x0, bb.y0, bb.x1, bb.y1))
        } else {
            print(String(format: "  %-12@  (not found)", layer.rawValue as NSString))
        }
    }

    let visURL = outDir.appendingPathComponent("components.png")
    try ComponentVisualizer.writeVisualization(flatLabeling, width: W, height: H, url: visURL)
    print("")
    print("  visual: \(visURL.path)")

    var layerMasks: [String: [UInt8]] = [:]
    for layer in ordered {
        guard let compID = assignment.layerComponent[layer] else { continue }
        let src = assignment.layerSource[layer] ?? .flatComponent
        let labeling = (src == .edgeComponent) ? edgeLabeling : flatLabeling

        var mask = [UInt8](repeating: 0, count: W * H)
        for i in 0..<(W * H) where labeling.componentID[i] == Int32(compID) {
            mask[i] = 1
        }
        layerMasks[layer.rawValue] = mask

        var rgba = [Float](repeating: 0, count: W * H * 4)
        for i in 0..<(W * H) {
            if mask[i] == 1 {
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

    let tOcc = Date()
    var typedMasks: [LayerID: [UInt8]] = [:]
    for (k, v) in layerMasks {
        if let lid = LayerID(rawValue: k) { typedMasks[lid] = v }
    }
    let occBuilder = OcclusionGraphBuilder()
    let occGraph = occBuilder.build(layerMasks: typedMasks, width: W, height: H)
    let envBuilder = MovementEnvelopeBuilder(budget: 20)
    let envelope = envBuilder.build(layerMasks: typedMasks, width: W, height: H)
    let dtOcc = Date().timeIntervalSince(tOcc)

    print("")
    print("== Occlusion Graph ==")
    print(String(format: "  layer order   : %@",
                 occGraph.orderedLayers.map { $0.rawValue }.joined(separator: " < ") as NSString))
    var ownerCounts: [Int] = Array(repeating: 0, count: occGraph.orderedLayers.count)
    var unowned = 0
    for r in occGraph.pixelOwnerRank {
        if r < 0 { unowned += 1 } else { ownerCounts[Int(r)] += 1 }
    }
    print("  pixelOwner distribution:")
    for (i, l) in occGraph.orderedLayers.enumerated() {
        print(String(format: "    %-12@  %7d px", l.rawValue as NSString, ownerCounts[i]))
    }
    print(String(format: "    (unowned)     %7d px", unowned))

    print("")
    print("== Movement Envelope (budget=20px) ==")
    print("  layer         layerPx   envelopePx  disocclusionBudget")
    let sortedEnv = envelope.layers.sorted { $0.value.layerPixels > $1.value.layerPixels }
    for (layer, e) in sortedEnv {
        print(String(format: "  %-12@  %7d   %9d   %9d",
                     layer.rawValue as NSString, e.layerPixels, e.envelopePixels, e.disocclusionBudget))
    }

    var compIDs: [String: Int] = [:]
    var sizes:   [String: Int] = [:]
    var srcMap:  [String: String] = [:]
    for (layer, compID) in assignment.layerComponent {
        compIDs[layer.rawValue] = compID
        sizes[layer.rawValue] = assignment.layerPixels[layer] ?? 0
        srcMap[layer.rawValue] = assignment.layerSource[layer]?.rawValue ?? "?"
    }

    let doc = XLDModelDocument(
        version: XLDKit.version,
        imagePath: url.lastPathComponent,
        imageWidth: W,
        imageHeight: H,
        layerComponentID: compIDs,
        layerPixelCount: sizes,
        layerSource: srcMap,
        layerMasks: layerMasks,
        occlusionGraph: occGraph,
        movementEnvelope: envelope,
        lifetimeDisocclusion: nil,
        lifetimeSequenceName: nil,
        lifetimePoseCount: 0
    )

    let modelDir = outDir.appendingPathComponent("model.xldmodel")
    try XLDModelIO.write(doc, to: modelDir)
    print("")
    print("  wrote \(modelDir.lastPathComponent)/")
    print("    manifest.json")
    print("    masks/ (\(layerMasks.count) PNG)")
    print("    README.txt")

    print("")
    print(String(format: "  gpu pipeline: %.2f ms", dtGPU * 1000))
    print(String(format: "  occlusion   : %.2f ms", dtOcc * 1000))
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

func cmdDepth(_ path: String) throws {
    let url = URL(fileURLWithPath: path)
    let outDir = url.deletingLastPathComponent().appendingPathComponent(
        url.deletingPathExtension().lastPathComponent + "_depth")
    try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    print("== XLD Depth Diagnostic ==")
    let img = try ImageIO.load(url: url)
    let W = img.width, H = img.height
    print("image: \(url.lastPathComponent)  \(W) x \(H)")

    let providers: [DepthEstimator] = [
        DepthStubProvider(),
        DepthAnythingV2Provider(),
    ]

    var any = false
    for pr in providers {
        print("")
        print("provider: \(pr.name) (\(pr.version))")
        let t0 = Date()
        guard let d = pr.estimate(image: img) else {
            print("  result: nil (unavailable)")
            continue
        }
        let dt = Date().timeIntervalSince(t0)
        any = true

        var dmin: UInt8 = 255, dmax: UInt8 = 0
        var dsum: UInt64 = 0
        var csum: UInt64 = 0
        for i in 0..<(W * H) {
            let v = d.depth[i]
            if v < dmin { dmin = v }
            if v > dmax { dmax = v }
            dsum += UInt64(v)
            csum += UInt64(d.confidence[i])
        }
        let dmean = Double(dsum) / Double(W * H)
        let cmean = Double(csum) / Double(W * H)
        print(String(format: "  depth: min=%d max=%d mean=%.1f", dmin, dmax, dmean))
        print(String(format: "  confidence mean=%.1f", cmean))
        print(String(format: "  time: %.1f ms", dt * 1000))

        let slug = pr.name.lowercased()
        let depthURL = outDir.appendingPathComponent("depth_\(slug).png")
        let colorURL = outDir.appendingPathComponent("depth_color_\(slug).png")
        let confURL  = outDir.appendingPathComponent("depth_confidence_\(slug).png")
        try DepthVisualizer.writeDepth(d, url: depthURL)
        try DepthVisualizer.writeDepthColor(d, url: colorURL)
        try DepthVisualizer.writeConfidence(d, url: confURL)
        print("  \(depthURL.path)")
        print("  \(colorURL.path)")
    }

    // offline structural depth (edge density based)
    if let gpu = MetalPipeline(),
       let labels = gpu.classify(image: img), labels.count == W * H {
        var edge = [UInt8](repeating: 0, count: W * H)
        for i in 0..<(W * H) where labels[i].classID == 1 { edge[i] = 1 }
        let bg = BackgroundRemover().remove(image: img)
        let t = Date()
        let sd = StructuralDepthEstimator()
            .estimate(image: img, edgeMask: edge, characterMask: bg.characterMask)
        let dt = Date().timeIntervalSince(t)
        any = true
        print("")
        print("provider: \(sd.providerName) (\(sd.providerVersion))")
        var dmin: UInt8 = 255, dmax: UInt8 = 0
        var dsum: UInt64 = 0
        for v in sd.depth {
            if v < dmin { dmin = v }
            if v > dmax { dmax = v }
            dsum += UInt64(v)
        }
        print(String(format: "  depth: min=%d max=%d mean=%.1f", dmin, dmax,
                     Double(dsum) / Double(W * H)))
        print(String(format: "  time: %.1f ms", dt * 1000))
        let dURL = outDir.appendingPathComponent("depth_structuraldensity.png")
        let cURL = outDir.appendingPathComponent("depth_color_structuraldensity.png")
        try DepthVisualizer.writeDepth(sd, url: dURL)
        try DepthVisualizer.writeDepthColor(sd, url: cURL)
        print("  \(dURL.path)")
        print("  \(cURL.path)")
    }

    if !any {
        print("")
        print("no depth provider produced output.")
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
    switch args.first {
    case "version", "--version", "-v":
        cmdVersion()
    case "pipeline":
        cmdPipeline()
    case "make-sample":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdMakeSample(args[1])
    case "make-sample-big":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdMakeSampleBig(args[1])
    case "run":
        guard args.count >= 2 else { usage(); exit(1) }
        var mode = "auto"
        var pathArg: String? = nil
        for a in args.dropFirst() {
            if a == "--gpu" { mode = "gpu" }
            else if a == "--cpu" { mode = "cpu" }
            else if a == "--auto" { mode = "auto" }
            else if pathArg == nil { pathArg = a }
        }
        guard let p = pathArg else { usage(); exit(1) }
        if mode == "auto" { mode = (MetalRuntime.shared != nil) ? "gpu" : "cpu" }
        try cmdRun(p, mode: mode)
    case "bench":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdBench(args[1])
    case "bench-gpu":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdBenchGPU(args[1])
    case "bench-sim":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdBenchSim(args[1])
    case "bench-scaling":
        try cmdBenchScaling()
    case "layers":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdLayers(args[1])
    case "move":
        guard args.count >= 5 else { usage(); exit(1) }
        guard let dx = Int(args[3]), let dy = Int(args[4]) else { usage(); exit(1) }
        try cmdMove(args[1], layer: args[2], dx: dx, dy: dy)
    case "simulate":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdSimulate(args[1])
    case "inpaint":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdInpaint(args[1])
    case "pipeline-full":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdPipelineFull(args[1])
    case "diagnose":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdDiagnose(args[1])
    case "depth":
        guard args.count >= 2 else { usage(); exit(1) }
        try cmdDepth(args[1])
    default:
        usage()
    }
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
