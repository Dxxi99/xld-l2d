# XLD Motion-First 2.5D — 비전 (2026-09-13)

> "진짜 3D를 복원한다"가 아니라
> "2D 그림이 특정 움직임에서 가장 자연스럽게 보이도록 가상의 구조를 만든다"

## 핵심 원칙
1. Depth는 목적이 아니라 변수
2. Raw → Landmark → Feature → Parameter → Deform
3. Tracking / Sim / Render 분리
4. Tracking Rig ≠ Model Rig (retarget)

## Motion Field Rigging
그림 전체에 "어디로 얼마나 움직일 수 있는가"를 field로 생성.
XLD 현재: basis_x/y.bin (global 계수) → layer별 계수로 확장.

## 통합 아키텍처 (최종 목표)
Tracking → Landmarks → Features → Retargeting
  → Motion Field → Motion-aware Depth
  → Occlusion/Disocclusion → Inpainting
  → Mesh Deform + Physics → 2.5D Render → Naturalness Test

## 부위별 파라미터 (목표)
머리 Head_XYZ / 눈 EyeOpen_LR / 동공 Pupil_XY / 눈썹 Brow
입 Mouth_Open/Form / 볼 Cheek / 턱 Jaw / 목 Neck
어깨 Shoulder / 팔 Arm joints / 손 Finger joints
몸통 Body pose / 하체 Leg joints

## 로드맵
- Phase 1: 눈/입 임시 Deformer (단일 mesh, 수동 bbox)
- Phase 2: Layer Mesh (7 layer 분리)
- Phase 3: Per-Region Motion Field (basis × layer 계수)
- Phase 4: Body Skeleton
- Phase 5: Motion-First Depth (역최적화)

## 절대 원칙
- Raw 데이터 보존
- Prototype → Real 교체 가능 구조
- 캐릭터 무관 (하드코딩 금지)
- 원본 최대 보존

저장일: 2026-09-13
