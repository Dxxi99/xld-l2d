# Architecture

## Three-layer unit model

| Layer            | Meaning                                  |
|------------------|------------------------------------------|
| Atomic Unit      | 1 pixel — isolation/cache/undo 단위       |
| Inference Unit   | Context Block — AI가 실제로 보는 문맥 블록 |
| Decision Unit    | 1 pixel — Accept/Reject/복원 단위         |

## Pipeline

    Illustration
      → Per-pixel Segmentation
      → Context Extraction
      → AI Inference (parallel, tiled)
      → Per-pixel Validation
      → Error Pixel Isolation → Re-inference
      → Layer Assembly
      → Occlusion Graph
      → Movement Envelope
      → Disocclusion Pixel Set
      → Minimal Inpainting (Manga + LaMa, parallel)
      → Fusion
      → Restoration Texture
      → Mesh → Deformation → Rig → Physics → 2.5D → Preview → Export

## Optimization Engine

모든 Stage가 다음 계층을 공유한다.

- TaskScheduler
- DependencyGraph
- AICache / TextureCache / MeshCache
- DirtyRegionManager
- MemoryManager
- CPU/GPU Scheduler
- ParallelExecutor
