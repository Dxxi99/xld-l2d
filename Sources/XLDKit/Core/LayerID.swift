import Foundation

public enum LayerID: String, CaseIterable, Sendable, Codable {
    case head, face
    case eyeL = "eye_L", eyeR = "eye_R"
    case eyebrowL = "eyebrow_L", eyebrowR = "eyebrow_R"
    case mouth
    case hairFront = "hair_front"
    case hairBack  = "hair_back"
    case body, neck, clothes
    case armL = "arm_L", armR = "arm_R"
    case background
}
