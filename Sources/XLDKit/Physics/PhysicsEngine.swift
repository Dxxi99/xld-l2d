import Foundation

public final class PhysicsEngine: @unchecked Sendable {
    public init() {}
    /// 실제 구현: hair / cloth 그룹별 시뮬레이션.
    /// 움직임이 없는 Group은 갱신을 건너뛰거나 캐시를 재사용.
    public func step(dirty: PixelMask?) {
        // Placeholder
    }
}
