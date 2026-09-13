#!/usr/bin/env python3
"""build_model.py — 일러스트 1장 → VTuber 모델 디렉토리 완전 자동 생성

Usage:
    python3 build_model.py <input.png> <out_model_dir> [work_dir]

Arguments:
    input.png        원본 일러스트 (흰 배경 PNG)
    out_model_dir    최종 결과 (Swift 앱이 로드할 model_vN/)
    work_dir         중간 산출물 (기본: <out_model_dir>_work)

Pipeline:
    [0-A] upscale_anime.py         -> upscaled.png (4x, Real-ESRGAN anime)
    [0-B] segment_toonout.py       -> char_mask.png (ToonOut)
    [1] test_depth.py              -> depth.png (on upscaled)
    [2] tj_normalized.py           -> tj_normalized_accept.npz (on upscaled)
    [3] layer_promotion_v2.py      -> layers_v2/, swift_cache/ (char_mask 재사용)
    [4] mesh_rig_v3.py             -> rig_v3.npz
    [5] mesh_rig_v8.py             -> rig_v8_sig120_R100.npz
    [6] mesh_rig_v12.py            -> rig_v12_flipfree.npz
    [7] cache_model_for_swift.py   -> model_dir/ (final)
"""
import sys
import subprocess
import time
from pathlib import Path

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

INPUT = Path(sys.argv[1]).expanduser().resolve()
OUT_MODEL = Path(sys.argv[2]).expanduser().resolve()
WORK = Path(sys.argv[3]).expanduser().resolve() if len(sys.argv) > 3 else (OUT_MODEL.parent / (OUT_MODEL.name + "_work"))

if not INPUT.exists():
    print(f"[error] input not found: {INPUT}")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
WORK.mkdir(parents=True, exist_ok=True)
OUT_MODEL.mkdir(parents=True, exist_ok=True)

print(f"[build] INPUT    = {INPUT}")
print(f"[build] OUT_MODEL= {OUT_MODEL}")
print(f"[build] WORK     = {WORK}")
print()

def run(label, cmd):
    print("=" * 70)
    print(f"[step] {label}")
    print("=" * 70)
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"[error] step failed: {label}")
        sys.exit(1)
    print(f"[ok] {label}  ({dt:.1f}s)")
    print()

# paths
UPSCALED_PNG = WORK / "upscaled.png"
CHAR_MASK    = WORK / "char_mask.png"
DEPTH_PNG   = WORK / "depth.png"
TJ_DIR      = WORK / "tj_normalized"
TJ_NPZ      = TJ_DIR / "tj_normalized_accept.npz"
LAYERS_DIR  = WORK / "layers_v2"
SWIFT_DIR   = WORK / "swift_cache"
LAYER_ID    = SWIFT_DIR / "expected_layer_id.bin"
META_TXT    = SWIFT_DIR / "meta.txt"
RIG_V3      = WORK / "rig_v3.npz"
RIG_V8      = WORK / "rig_v8_sig120_R100.npz"
RIG_V12     = WORK / "rig_v12_flipfree.npz"

# 0-A) upscale (Real-ESRGAN anime, 4x)
run("upscale (Real-ESRGAN anime)", [
    str(SCRIPT_DIR / "upscale_anime.py"),
    str(INPUT), str(UPSCALED_PNG)
])

# 0-B) segmentation (ToonOut, BiRefNet anime)
run("segment (ToonOut)", [
    str(SCRIPT_DIR / "segment_toonout.py"),
    str(UPSCALED_PNG), str(CHAR_MASK)
])

# 1) depth (on upscaled image)
run("depth inference", [
    str(SCRIPT_DIR / "test_depth.py"),
    str(UPSCALED_PNG), str(DEPTH_PNG)
])

# 2) T-junction (on upscaled image)
run("T-junction detection", [
    str(SCRIPT_DIR / "tj_normalized.py"),
    str(UPSCALED_PNG), str(TJ_DIR)
])

# 3) layer promotion (fusion + cache) — char_mask 재사용
run("layer promotion + fusion", [
    str(SCRIPT_DIR / "layer_promotion_v2.py"),
    str(UPSCALED_PNG), str(DEPTH_PNG), str(TJ_NPZ),
    str(LAYERS_DIR), str(SWIFT_DIR),
    str(CHAR_MASK)
])

# 4) rig v3
run("rig v3 (smooth sign + basis)", [
    str(SCRIPT_DIR / "mesh_rig_v3.py"),
    str(LAYER_ID), str(META_TXT), str(RIG_V3), "12"
])

# 5) rig v8
run("rig v8 (gauss-seidel Lipschitz)", [
    str(SCRIPT_DIR / "mesh_rig_v8.py"),
    str(RIG_V3), str(DEPTH_PNG), str(RIG_V8),
    "120", "20", "100", "0.50"
])

# 6) rig v12 (flip-free)
run("rig v12 (flip-free)", [
    str(SCRIPT_DIR / "mesh_rig_v12.py"),
    str(RIG_V8), str(RIG_V12), "0.15", "3000"
])

# 7) cache for Swift (final model dir)
run("cache for Swift (model dir)", [
    str(SCRIPT_DIR / "cache_model_for_swift.py"),
    str(RIG_V12), str(UPSCALED_PNG), str(OUT_MODEL), str(SWIFT_DIR),
    str(DEPTH_PNG)
])

# 8) copy downsampled depth for 2.5D shader
run("copy depth (2.5D)", [
    str(SCRIPT_DIR / "copy_depth_for_2p5d.py"),
    str(DEPTH_PNG), str(OUT_MODEL)
])

# 9) anime face regions → meta.txt (자동 눈/입 좌표)
run("face regions (auto)", [
    str(SCRIPT_DIR / "face_regions.py"),
    str(UPSCALED_PNG), str(OUT_MODEL)
])

print("=" * 70)
print(f"[done] model ready: {OUT_MODEL}")
print("=" * 70)
for f in sorted(OUT_MODEL.iterdir()):
    print(f"  {f.name}  {f.stat().st_size} bytes")
print()
print(f"[next] add to app: file://{OUT_MODEL}")
