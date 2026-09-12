import Foundation

enum ModelLoaderError: Error, LocalizedError {
    case pythonNotFound(URL)
    case scriptNotFound(URL)
    case processFailed(Int32, String)
    case outputDirMissing(URL)

    var errorDescription: String? {
        switch self {
        case .pythonNotFound(let u):  return "Python not found: \(u.path)"
        case .scriptNotFound(let u):  return "Script not found: \(u.path)"
        case .processFailed(let c, let m): return "Process failed (\(c)): \(m.prefix(300))"
        case .outputDirMissing(let u): return "Output dir not created: \(u.path)"
        }
    }
}

enum ModelLoader {

    static let repoRoot: URL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("XLD-L2D")

    static var pythonPath: URL { repoRoot.appendingPathComponent(".venv/bin/python3") }
    static var buildScript: URL { repoRoot.appendingPathComponent("tools/scripts/build_model.py") }

    static func runPipeline(inputImage: URL,
                            outModelDir: URL,
                            progress: @escaping (Int, Int, String) -> Void) throws {

        print("[ModelLoader] START input=\(inputImage.path)")
        print("[ModelLoader] OUT=\(outModelDir.path)")

        guard FileManager.default.fileExists(atPath: pythonPath.path) else {
            throw ModelLoaderError.pythonNotFound(pythonPath)
        }
        guard FileManager.default.fileExists(atPath: buildScript.path) else {
            throw ModelLoaderError.scriptNotFound(buildScript)
        }

        try? FileManager.default.createDirectory(at: outModelDir, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = pythonPath
        process.arguments = [buildScript.path, inputImage.path, outModelDir.path]
        process.currentDirectoryURL = repoRoot

        // Force unbuffered stdout/stderr
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        let handle = outPipe.fileHandleForReading
        var buffer = Data()
        var stepCounter = 0

        let stepLabels: [String: String] = [
            "depth inference": "Depth inference",
            "T-junction detection": "T-junction detection",
            "layer promotion + fusion": "Layer fusion",
            "rig v3": "Rig v3 (smooth sign)",
            "rig v8": "Rig v8 (Lipschitz)",
            "rig v12": "Rig v12 (flip-free)",
            "cache for Swift": "Cache for Swift"
        ]

        handle.readabilityHandler = { fh in
            let chunk = fh.availableData
            if chunk.isEmpty {
                fh.readabilityHandler = nil
                return
            }
            buffer.append(chunk)
            // split on newline
            while let nl = buffer.firstIndex(of: 0x0A) {
                let lineData = buffer.subdata(in: 0..<nl)
                buffer.removeSubrange(0...nl)
                guard let line = String(data: lineData, encoding: .utf8) else { continue }
                print("[ModelLoader] \(line)")
                if line.hasPrefix("[step] ") {
                    stepCounter += 1
                    let label = String(line.dropFirst("[step] ".count))
                    let nice = stepLabels[label] ?? label
                    progress(stepCounter, 7, nice)
                }
            }
        }

        do {
            try process.run()
            print("[ModelLoader] pid=\(process.processIdentifier)")
        } catch {
            print("[ModelLoader] run() failed: \(error)")
            throw error
        }

        process.waitUntilExit()
        handle.readabilityHandler = nil

        // drain stderr for error reporting
        let errData = (try? errPipe.fileHandleForReading.readToEnd()) ?? Data()
        let errText = String(data: errData, encoding: .utf8) ?? ""
        if !errText.isEmpty {
            print("[ModelLoader] stderr:\n\(errText)")
        }

        print("[ModelLoader] exit=\(process.terminationStatus)")

        if process.terminationStatus != 0 {
            throw ModelLoaderError.processFailed(process.terminationStatus, errText)
        }
    }
}
