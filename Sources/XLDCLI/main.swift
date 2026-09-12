import Foundation
import XLDKit

let args = Array(CommandLine.arguments.dropFirst())

func printBanner() {
    print("XLD L2D — \(XLDKit.version)")
    print("Pipeline version: \(XLDKit.pipelineVersion)")
    print("")
}

func cmdVersion() {
    printBanner()
}

func cmdPipeline() {
    printBanner()
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

switch args.first {
case "version", "--version", "-v":
    cmdVersion()
case "pipeline", nil:
    cmdPipeline()
default:
    print("usage: xld [version|pipeline]")
}
