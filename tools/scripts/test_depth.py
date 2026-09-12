#!/usr/bin/env python3
"""test_depth.py — Depth Anything V2 CoreML inference

Usage:
    python3 test_depth.py [IMG_png] [OUT_png] [MODEL_dir]

Defaults:
    IMG_png   = assets/real/my_vtuber.png
    OUT_png   = assets/real/depth_anything_v2.png
    MODEL_dir = models/depth-anything-v2-small

Character bbox is auto-detected via border-connected light-pixel mask.
"""
import sys, time
import numpy as np
import coremltools as ct
from PIL import Image
from pathlib import Path
from scipy import ndimage

# ---- CLI ----
IMG_ARG   = sys.argv[1] if len(sys.argv) > 1 else None
OUT_ARG   = sys.argv[2] if len(sys.argv) > 2 else None
MODEL_ARG = sys.argv[3] if len(sys.argv) > 3 else None

IMG_PATH   = Path(IMG_ARG).expanduser()   if IMG_ARG   else (Path.home()/"XLD-L2D/assets/real/my_vtuber.png")
OUT_PATH   = Path(OUT_ARG).expanduser()   if OUT_ARG   else (Path.home()/"XLD-L2D/assets/real/depth_anything_v2.png")
MODEL_DIR  = Path(MODEL_ARG).expanduser() if MODEL_ARG else (Path.home()/"XLD-L2D/models/depth-anything-v2-small")

MODEL_PKG = MODEL_DIR / "DepthAnythingV2SmallF32.mlpackage"
if not IMG_PATH.exists():   print(f"[error] img: {IMG_PATH}");   sys.exit(1)
if not MODEL_PKG.exists():  print(f"[error] model: {MODEL_PKG}"); sys.exit(1)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"[cfg] IMG={IMG_PATH}")
print(f"[cfg] OUT={OUT_PATH}")
print(f"[cfg] MODEL={MODEL_PKG}")

# ---- load model ----
model = ct.models.MLModel(str(MODEL_PKG))
spec = model.get_spec()
inp = spec.description.input[0]
input_name = inp.name
W = inp.type.imageType.width
H = inp.type.imageType.height
print(f"[model] input {W}x{H}")

# ---- load image ----
img = Image.open(IMG_PATH).convert("RGB")
I_W, I_H = img.size
print(f"[img] {I_W}x{I_H}")
rgb = np.array(img)

# ---- auto char bbox (border-connected light = bg) ----
border = np.concatenate([rgb[0,:], rgb[-1,:], rgb[:,0], rgb[:,-1]], 0)
bg = np.median(border, 0)
diff = np.linalg.norm(rgb.astype(np.float32) - bg, axis=-1)
similar = diff < 40
lbl, _ = ndimage.label(similar)
bl = np.concatenate([lbl[0,:], lbl[-1,:], lbl[:,0], lbl[:,-1]]); bl = bl[bl>0]
if len(bl) > 0:
    bg_label = np.bincount(bl).argmax()
    cm = ~(lbl == bg_label)
else:
    cm = np.ones_like(similar)
# bbox from char mask
ys, xs = np.where(cm)
if len(ys) == 0:
    print("[warn] char mask empty, using full image")
    x0, y0, x1, y1 = 0, 0, I_W, I_H
else:
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
print(f"[bbox] char=({x0},{y0})-({x1},{y1})  size={x1-x0}x{y1-y0}")

# ---- crop + letterbox ----
crop = img.crop((x0, y0, x1, y1))
scale = min(W / crop.width, H / crop.height)
nw = int(round(crop.width  * scale))
nh = int(round(crop.height * scale))
resized = crop.resize((nw, nh), Image.LANCZOS)
canvas = Image.new("RGB", (W, H), (0, 0, 0))
ox = (W - nw) // 2
oy = (H - nh) // 2
canvas.paste(resized, (ox, oy))
print(f"[letterbox] {nw}x{nh} at ({ox},{oy}) scale={scale:.3f}")

# ---- inference ----
t0 = time.time()
out = model.predict({input_name: canvas})
dt = time.time() - t0
print(f"[inference] {dt*1000:.1f} ms")

raw = list(out.values())[0]
depth = np.array(raw).astype(np.float32)
if depth.ndim == 3: depth = depth[..., 0]
print(f"[depth_raw] {depth.shape} min/max/mean {depth.min():.3f}/{depth.max():.3f}/{depth.mean():.3f}")

# ---- crop letterbox padding ----
depth_crop = depth[oy:oy+nh, ox:ox+nw]

# ---- normalize ----
d = (depth_crop - depth_crop.min()) / (depth_crop.max() - depth_crop.min() + 1e-8)
d8 = (d * 255).astype(np.uint8)

# ---- resize back to char bbox size ----
d_char = Image.fromarray(d8).resize((x1-x0, y1-y0), Image.BILINEAR)

# ---- paste into full canvas ----
full = Image.new("L", (I_W, I_H), 0)
full.paste(d_char, (x0, y0))
full.save(OUT_PATH)
print(f"[save] {OUT_PATH}  {full.size}")

# ---- stats ----
arr = np.array(full)
char_pixels = arr[cm]
if len(char_pixels) > 0:
    print(f"[stats] char depth: min {char_pixels.min()} max {char_pixels.max()} mean {char_pixels.mean():.1f} std {char_pixels.std():.1f}")
