#!/usr/bin/env python3
"""build_model.py — 일러스트 1장 → VTuber 모델 디렉토리 완전 자동 생성

Usage:
    python3 build_model.py <input.png> <out_model_dir> [work_dir]

Arguments:
    input.png        원본 일러스트 (흰 배경 PNG)
    out_model_dir    최종 결과 (Swift 앱이 로드할 model_vN/)
    work_dir         중간 산출물 (기본: <out_model_dir>_work)

Pipeline:
    [1] test_depth.py              -> depth.png
    [2] tj_normalized.py           -> tj_normalized_accept.npz
    [3] layer_promotion_v2.py      -> layers_v2/, swift_cache/ (layer_id, cons, seg)
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

# 1) depth
run("depth inference", [
    str(SCRIPT_DIR / "test_depth.py"),
    str(INPUT), str(DEPTH_PNG)
])

# 2) T-junction
run("T-junction detection", [
    str(SCRIPT_DIR / "tj_normalized.py"),
    str(INPUT), str(TJ_DIR)
])

# 3) layer promotion (fusion + cache)
run("layer promotion + fusion", [
    str(SCRIPT_DIR / "layer_promotion_v2.py"),
    str(INPUT), str(DEPTH_PNG), str(TJ_NPZ),
    str(LAYERS_DIR), str(SWIFT_DIR)
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
    str(RIG_V12), str(INPUT), str(OUT_MODEL)
])

print("=" * 70)
print(f"[done] model ready: {OUT_MODEL}")
print("=" * 70)
for f in sorted(OUT_MODEL.iterdir()):
    print(f"  {f.name}  {f.stat().st_size} bytes")
print()
print(f"[next] add to app: file://{OUT_MODEL}")
