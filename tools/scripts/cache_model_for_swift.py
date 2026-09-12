#!/usr/bin/env python3
"""cache_model_for_swift.py — rig_v12 + source img → Swift model dir

Usage:
    python3 cache_model_for_swift.py [SRC_npz] [SRC_img_png] [DST_dir]

Defaults:
    SRC_npz     = assets/real/mesh/rig_v12_flipfree.npz
    SRC_img_png = assets/real/my_vtuber.png
    DST_dir     = assets/real/swift_cache/model_v1

Also performs white-background flood fill → alpha=0 on source.png.
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage

# ---- CLI ----
SRC_ARG = sys.argv[1] if len(sys.argv) > 1 else None
IMG_ARG = sys.argv[2] if len(sys.argv) > 2 else None
DST_ARG = sys.argv[3] if len(sys.argv) > 3 else None

SRC = Path(SRC_ARG).expanduser() if SRC_ARG else (
    Path.home() / "XLD-L2D/assets/real/mesh/rig_v12_flipfree.npz")
IMG = Path(IMG_ARG).expanduser() if IMG_ARG else (
    Path.home() / "XLD-L2D/assets/real/my_vtuber.png")
DST = Path(DST_ARG).expanduser() if DST_ARG else (
    Path.home() / "XLD-L2D/assets/real/swift_cache/model_v1")

if not SRC.exists(): print(f"[error] npz: {SRC}"); sys.exit(1)
if not IMG.exists(): print(f"[error] img: {IMG}"); sys.exit(1)
DST.mkdir(parents=True, exist_ok=True)

print(f"[cfg] SRC={SRC}")
print(f"[cfg] IMG={IMG}")
print(f"[cfg] DST={DST}")

# ---- load rig ----
d = np.load(SRC)
xs = d['xs'].astype(np.float32)
ys = d['ys'].astype(np.float32)
tris = d['triangles'].astype(np.int32)
bx = d['basis_x'].astype(np.float32)
by = d['basis_y'].astype(np.float32)
H, W = int(d['H']), int(d['W'])
R_X = float(d['R_X']); R_Y = float(d['R_Y'])
n_v = len(xs); n_t = len(tris)
print(f"[rig] n_v={n_v} n_t={n_t} H={H} W={W} R_X={R_X} R_Y={R_Y}")

# ---- dump vertex/tri/basis ----
verts_xy = np.stack([xs, ys], axis=1).astype(np.float32)
verts_xy.tofile(DST / "vertices_xy.bin")
tris.astype(np.int32).tofile(DST / "triangles.bin")
bx.tofile(DST / "basis_x.bin")
by.tofile(DST / "basis_y.bin")

# ---- layer_id per vertex (optional) ----
if 'v_layer' in d.files:
    v_layer = d['v_layer'].astype(np.uint8)
    v_layer.tofile(DST / "layer_id.bin")
    print(f"[layer] unique={np.unique(v_layer).tolist()}")

# ---- source image + white bg flood fill → alpha=0 ----
img = Image.open(IMG).convert("RGBA")
if img.size != (W, H):
    img = img.resize((W, H), Image.LANCZOS)

a = np.array(img).copy()
rgb = a[:,:,:3].astype(np.int32)
white = (rgb[:,:,0] > 235) & (rgb[:,:,1] > 235) & (rgb[:,:,2] > 235)
lbl, n = ndimage.label(white)
border_labels = set()
border_labels.update(np.unique(lbl[0, :]).tolist())
border_labels.update(np.unique(lbl[-1, :]).tolist())
border_labels.update(np.unique(lbl[:, 0]).tolist())
border_labels.update(np.unique(lbl[:, -1]).tolist())
border_labels.discard(0)
bg = np.isin(lbl, list(border_labels))
a[:,:,3] = np.where(bg, 0, 255).astype(np.uint8)
Image.fromarray(a, "RGBA").save(DST / "source.png")
print(f"[img] source.png {img.size}  bg_removed={int(bg.sum())} px ({100*bg.mean():.1f}%)")

# ---- meta ----
with open(DST / "meta.txt", "w") as f:
    f.write(f"n_v {n_v}\n")
    f.write(f"n_t {n_t}\n")
    f.write(f"R_X {R_X}\n")
    f.write(f"R_Y {R_Y}\n")
    f.write(f"H {H}\n")
    f.write(f"W {W}\n")

print(f"[save] {DST}")
for f in sorted(DST.iterdir()):
    print(f"  {f.name}  {f.stat().st_size} bytes")
