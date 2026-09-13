#!/usr/bin/env python3
"""segment_toonout.py — ToonOut (BiRefNet anime fine-tune) ONNX 누끼

Usage:
    python3 segment_toonout.py <input.png> <out_mask.png> [--onnx PATH] [--size 1024]

동작:
    1. 이미지 로드
    2. size로 리사이즈 (기본 1024, ToonOut 최적)
    3. ONNX Runtime 추론 → 마스크 (0~1)
    4. 원본 크기로 LANCZOS 리사이즈
    5. 저장 (grayscale PNG)
"""
import os
# ONNX Runtime 텔레메트리 종료 크래시 방지
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["ORT_TELEMETRY_DISABLED"] = "1"

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

DEFAULT_ONNX = "models/toonout/birefnet-toonout-fp16.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    inp = Path(args.input).expanduser().resolve()
    outp = Path(args.output).expanduser().resolve()
    mp = Path(args.onnx)
    if not mp.is_absolute():
        mp = (Path(__file__).resolve().parent.parent.parent / mp).resolve()

    if not inp.exists():
        print(f"[error] input not found: {inp}")
        sys.exit(1)
    if not mp.exists():
        print(f"[error] onnx not found: {mp}")
        sys.exit(1)

    print(f"[load onnx] {mp}")
    opts = ort.SessionOptions()
    opts.enable_profiling = False
    # 그래프 최적화만 (불필요 로그 억제)
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(mp), sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    img = Image.open(inp).convert("RGB")
    W, H = img.size
    print(f"[input] {W}x{H}")

    # 전처리
    size = args.size
    img_r = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img_r, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    x = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    print(f"[tensor] {x.shape}")

    t0 = time.time()
    out = sess.run([out_name], {in_name: x})[0]
    dt = time.time() - t0
    print(f"[infer] {dt:.2f}s  out shape={out.shape}")

    mask = out[0, 0]  # (size, size)
    # logit이면 sigmoid
    if mask.min() < 0 or mask.max() > 1:
        mask = 1.0 / (1.0 + np.exp(-mask))
    print(f"[mask] range={mask.min():.3f}~{mask.max():.3f}  mean={mask.mean():.3f}")

    # 원본 크기로
    mask_full = np.array(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)
    ).astype(np.float32) / 255.0

    outp.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_full * 255).astype(np.uint8)).save(outp)
    print(f"[done] saved: {outp}  {mask_full.shape}")


if __name__ == "__main__":
    main()
