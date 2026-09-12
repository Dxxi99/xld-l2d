import Foundation
import Combine

/// Observable state for the rig viewer.
public final class RigState: ObservableObject {
    @Published public var yawDeg: Double = 0
    @Published public var pitchDeg: Double = 0
    @Published public var fps: Double = 0
    @Published public var statusText: String = "initializing..."

    public init() {}
}
