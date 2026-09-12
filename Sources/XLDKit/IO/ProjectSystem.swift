import Foundation

public final class ProjectSystem {
    public private(set) var model = XLDModel()
    public init() {}

    public func save(to url: URL) throws {
        // Placeholder: encode XLDModel + textures.
        try Data().write(to: url)
    }
}
