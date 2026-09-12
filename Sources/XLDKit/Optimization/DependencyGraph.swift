import Foundation

public final class DependencyGraph: @unchecked Sendable {
    private var deps: [String: Set<String>] = [:]
    public init() {}

    public func add(_ node: String, dependsOn: Set<String> = []) {
        deps[node, default: []].formUnion(dependsOn)
    }

    public func dependencies(of node: String) -> Set<String> {
        deps[node] ?? []
    }
}
