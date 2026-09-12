import Foundation

public enum PipelineStage: String, CaseIterable, Sendable {
    case analysis          = "Image Analysis"
    case segmentation      = "Per-Pixel Segmentation"
    case contextExtraction = "Context Extraction"
    case aiInference       = "AI Inference"
    case validation        = "Pixel Validation"
    case errorIsolation    = "Error Isolation"
    case layerAssembly     = "Layer Assembly"
    case occlusion         = "Occlusion Graph"
    case envelope          = "Movement Envelope"
    case disocclusion      = "Disocclusion"
    case inpainting        = "Minimal Inpainting"
    case fusion            = "Fusion"
    case restoration       = "Restoration Texture"
    case mesh              = "Mesh"
    case deformation       = "Deformation"
    case rig               = "Rig"
    case physics           = "Physics"
    case preview           = "Preview"
    case export            = "Export"
}
