#!/usr/bin/env python3
"""upscale_anime.py — Real-ESRGAN anime (CoreML) 타일 업스케일러"""
import sys, argparse, time
from pathlib import Path
import numpy as np
from PIL import Image
import coremltools as ct

ALLOWED_TILES = [256, 384, 512, 640, 768]
DEFAULT_MODEL = "models/Real-ESRGAN/RealESRGAN_anime.mlpackage"

def make_weight(tile, overlap):
    w = np.ones((tile, tile), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    w[:overlap, :] *= ramp[:, None]
    w[-overlap:, :] *= ramp[::-1, None]
    w[:, :overlap] *= ramp[None, :]
    w[:, -overlap:] *= ramp[::-1][None, :]
    return w

def upscale_tile(model, tile_rgb):
    x = tile_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    out = model.predict({"input": x})
    y = list(out.values())[0]
    y = np.transpose(y[0], (1, 2, 0))
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)

def upscale_image(model, rgb, tile, overlap, max_size):
    H, W = rgb.shape[:2]
    scale = 4
    if max(H, W) * scale > max_size:
        limit = max_size // scale
        if H >= W:
            newH, newW = limit, max(1, int(round(W * limit / H)))
        else:
            newW, newH = limit, max(1, int(round(H * limit / W)))
        print(f"[limit] {W}x{H} -> {newW}x{newH}")
        rgb = np.array(Image.fromarray(rgb).resize((newW, newH), Image.LANCZOS))
        H, W = rgb.shape[:2]
    outH, outW = H * scale, W * scale
    print(f"[upscale] {W}x{H} -> {outW}x{outH}")
    pad_h = (tile - (H % tile)) % tile
    pad_w = (tile - (W % tile)) % tile
    padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    PH, PW = padded.shape[:2]
    acc = np.zeros((outH + pad_h * scale, outW + pad_w * scale, 3), dtype=np.float32)
    wsum = np.zeros((outH + pad_h * scale, outW + pad_w * scale), dtype=np.float32)
    weight = make_weight(tile * scale, overlap * scale)
    step = tile - overlap
    ys = list(range(0, max(PH - tile, 0) + 1, step))
    if not ys or ys[-1] != PH - tile:
        ys.append(max(PH - tile, 0))
    xs = list(range(0, max(PW - tile, 0) + 1, step))
    if not xs or xs[-1] != PW - tile:
        xs.append(max(PW - tile, 0))
    total = len(ys) * len(xs); n = 0; t0 = time.time()
    for y in ys:
        for x in xs:
            up = upscale_tile(model, padded[y:y+tile, x:x+tile])
            oy, ox = y * scale, x * scale
            acc[oy:oy+tile*scale, ox:ox+tile*scale] += up.astype(np.float32) * weight[..., None]
            wsum[oy:oy+tile*scale, ox:ox+tile*scale] += weight
            n += 1
    wsum = np.maximum(wsum, 1e-6)
    result = acc / wsum[..., None]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result[:outH, :outW]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--max", type=int, default=16384)
    args = ap.parse_args()
    if args.tile not in ALLOWED_TILES:
        print(f"[error] tile must be one of {ALLOWED_TILES}"); sys.exit(1)
    inp = Path(args.input).expanduser().resolve()
    outp = Path(args.output).expanduser().resolve()
    mp = Path(args.model)
    if not mp.is_absolute():
        mp = (Path(__file__).resolve().parent.parent.parent / mp).resolve()
    if not inp.exists(): print(f"[error] input: {inp}"); sys.exit(1)
    if not mp.exists(): print(f"[error] model: {mp}"); sys.exit(1)
    print(f"[load] {mp}")
    model = ct.models.MLModel(str(mp), compute_units=ct.ComputeUnit.ALL)
    img = Image.open(inp).convert("RGBA")
    arr = np.array(img)
    rgb, alpha = arr[..., :3], arr[..., 3]
    print(f"[info] {img.size}")
    rgb_up = upscale_image(model, rgb, args.tile, args.overlap, args.max)
    H, W = rgb_up.shape[:2]
    a_up = np.array(Image.fromarray(alpha).resize((W, H), Image.LANCZOS))
    out_img = Image.fromarray(np.dstack([rgb_up, a_up]), mode="RGBA")
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(outp, "PNG")
    print(f"[done] {outp}  {out_img.size[0]}x{out_img.size[1]}")

if __name__ == "__main__":
    main()
