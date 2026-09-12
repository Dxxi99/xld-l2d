import Foundation

public enum ExportFormat: String, Sendable {
    case xldNative = "xld"
    case syphon    = "syphon"
    case jsonMeta  = "json"
}

public final class Exporter {
    public init() {}

    public func export(model: XLDModel, format: ExportFormat, to url: URL) throws {
        // Placeholder
    }
}
