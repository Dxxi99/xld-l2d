import Foundation

public struct XLDModelDocument: Sendable {
    public var version: String
    public var imagePath: String
    public var imageWidth: Int
    public var imageHeight: Int
    public var layerComponentID: [String: Int]
    public var layerPixelCount: [String: Int]
    public var layerSource: [String: String]
    public var layerMasks: [String: [UInt8]]
    public var occlusionGraph: OcclusionGraph?
    public var movementEnvelope: MovementEnvelope?

    public init(version: String,
                imagePath: String,
                imageWidth: Int,
                imageHeight: Int,
                layerComponentID: [String: Int],
                layerPixelCount: [String: Int],
                layerSource: [String: String],
                layerMasks: [String: [UInt8]],
                occlusionGraph: OcclusionGraph?,
                movementEnvelope: MovementEnvelope?) {
        self.version = version
        self.imagePath = imagePath
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.layerComponentID = layerComponentID
        self.layerPixelCount = layerPixelCount
        self.layerSource = layerSource
        self.layerMasks = layerMasks
        self.occlusionGraph = occlusionGraph
        self.movementEnvelope = movementEnvelope
    }
}

public enum XLDModelIO {
    public static func write(_ doc: XLDModelDocument, to dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        var manifest: [String: Any] = [:]
        manifest["version"] = doc.version
        manifest["imagePath"] = doc.imagePath
        manifest["imageWidth"] = doc.imageWidth
        manifest["imageHeight"] = doc.imageHeight
        manifest["layerComponentID"] = doc.layerComponentID
        manifest["layerPixelCount"] = doc.layerPixelCount
        manifest["layerSource"] = doc.layerSource

        if let graph = doc.occlusionGraph {
            var ordered: [String] = []
            for l in graph.orderedLayers { ordered.append(l.rawValue) }
            manifest["occlusionOrder"] = ordered
            var zmap: [String: Int] = [:]
            for (k, v) in graph.layerZ { zmap[k.rawValue] = v }
            manifest["occlusionZ"] = zmap

            let json = try JSONSerialization.data(withJSONObject: manifest,
                                                  options: [.prettyPrinted, .sortedKeys])
            try json.write(to: dir.appendingPathComponent("manifest.json"))

            var occlusionMeta: [String: Any] = [:]
            occlusionMeta["orderedLayers"] = ordered
            occlusionMeta["layerZ"] = zmap
            occlusionMeta["width"] = graph.width
            occlusionMeta["height"] = graph.height

            let occJSON = try JSONSerialization.data(withJSONObject: occlusionMeta,
                                                     options: [.prettyPrinted, .sortedKeys])
            try occJSON.write(to: dir.appendingPathComponent("occlusion.json"))

            let zonesURL = dir.appendingPathComponent("occlusion_zones.png")
            try OcclusionVisualizer.writeZones(graph, url: zonesURL)
        } else {
            let json = try JSONSerialization.data(withJSONObject: manifest,
                                                  options: [.prettyPrinted, .sortedKeys])
            try json.write(to: dir.appendingPathComponent("manifest.json"))
        }

        if let env = doc.movementEnvelope {
            var layerObj: [String: Any] = [:]
            for (layer, e) in env.layers {
                layerObj[layer.rawValue] = [
                    "bbox": [e.bboxX0, e.bboxY0, e.bboxX1, e.bboxY1],
                    "expanded": [e.expandedX0, e.expandedY0, e.expandedX1, e.expandedY1],
                    "layerPixels": e.layerPixels,
                    "envelopePixels": e.envelopePixels,
                    "disocclusionBudget": e.disocclusionBudget,
                ]
            }
            let envObj: [String: Any] = [
                "budget": env.budget,
                "layers": layerObj,
            ]
            let envJSON = try JSONSerialization.data(withJSONObject: envObj,
                                                     options: [.prettyPrinted, .sortedKeys])
            try envJSON.write(to: dir.appendingPathComponent("movement_envelope.json"))
        }

        let masksDir = dir.appendingPathComponent("masks")
        try FileManager.default.createDirectory(at: masksDir, withIntermediateDirectories: true)
        for (layer, mask) in doc.layerMasks {
            let total = doc.imageWidth * doc.imageHeight
            var rgba = [Float](repeating: 0, count: total * 4)
            for i in 0..<total {
                if mask[i] == 1 {
                    rgba[i * 4 + 0] = 1
                    rgba[i * 4 + 1] = 1
                    rgba[i * 4 + 2] = 1
                }
                rgba[i * 4 + 3] = 1.0
            }
            let img = RasterImage(width: doc.imageWidth, height: doc.imageHeight, rgba: rgba)
            let url = masksDir.appendingPathComponent("\(layer).png")
            try ImageIO.save(image: img, url: url)
        }

        var summary = """
        version=\(doc.version)
        image=\(doc.imagePath) (\(doc.imageWidth)x\(doc.imageHeight))
        layers=\(doc.layerMasks.count)
        """
        if let g = doc.occlusionGraph {
            summary += "\nocclusionOrder=\(g.orderedLayers.map { $0.rawValue }.joined(separator: " < "))"
        }
        if let e = doc.movementEnvelope {
            summary += "\nmovementBudget=\(e.budget)px"
            summary += "\nenvelopeLayers=\(e.layers.count)"
        }
        try summary.write(to: dir.appendingPathComponent("README.txt"),
                          atomically: true, encoding: .utf8)
    }
}
