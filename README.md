# XLD L2D

macOS-native 2D/2.5D VTuber model builder.

## Principle

> Atomic Unit = Pixel
> Inference Unit = Context Block
> Decision Unit = Pixel

> "가능한 가장 작은 단위로 정확하게 이해하고, 필요한 것만 계산하며,
>  원본은 최대한 보존하고, AI의 오류는 최소 영역에 격리한다."

## Layout

- `Sources/XLDKit/Core`         — PixelUnit, ContextBlock, PixelMask, LayerID
- `Sources/XLDKit/Pipeline`     — Stage 정의 및 실행기
- `Sources/XLDKit/Optimization` — TaskScheduler, AICache, DirtyRegion, DependencyGraph
- `Sources/XLDKit/AI`           — AI Runtime, Inpainting 모델 래퍼
- `Sources/XLDKit/Graphics`     — Metal Context, Renderer
- `Sources/XLDKit/Rig`          — Rig / Deformer
- `Sources/XLDKit/Physics`      — Hair / Cloth Physics
- `Sources/XLDKit/IO`           — XLD Model, Project, Export
- `Sources/XLDCLI`              — `xld` CLI
- `Shaders`                     — Metal compute shaders
- `ai`                          — Python AI research / training
- `docs`                        — Architecture docs

## Build

    ./build.sh

## Run

    swift run xld
    swift run xld pipeline
    swift run xld version
