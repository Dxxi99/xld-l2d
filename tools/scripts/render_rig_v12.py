#!/usr/bin/env python3
"""render_rig_v12.py — 수정: Y 뒤집기 제거"""
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

P = Path.home() / "XLD-L2D/assets/real/mesh/rig_v12_flipfree.npz"
d = np.load(P)
xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
bx = d['basis_x'].astype(np.float64)
by = d['basis_y'].astype(np.float64)
R_X = float(d['R_X']); R_Y = float(d['R_Y'])
H, W = int(d['H']), int(d['W'])
print(f"[load] n_v={len(xs)} n_t={len(tris)}  R_X={R_X} R_Y={R_Y}")

SRC_IMG = Path.home() / "XLD-L2D/assets/real/my_vtuber.png"
src = Image.open(SRC_IMG).convert("RGBA")
if src.size != (W, H):
    src = src.resize((W, H), Image.LANCZOS)
src_np = np.array(src)

tri_cx = (xs[tris[:,0]] + xs[tris[:,1]] + xs[tris[:,2]]) / 3
tri_cy = (ys[tris[:,0]] + ys[tris[:,1]] + ys[tris[:,2]]) / 3
ti = np.clip(tri_cx.astype(np.int32), 0, W-1)
tj = np.clip(tri_cy.astype(np.int32), 0, H-1)
tri_col = src_np[tj, ti, :3] / 255.0

POSES = [
    ("rest",       0,   0),
    ("yaw +25",   25,   0),
    ("yaw -25",  -25,   0),
    ("pitch +15",  0,  15),
    ("pitch -15",  0, -15),
    ("yaw+25 p+15",25,  15),
]

fig = plt.figure(figsize=(24, 16), dpi=90)
fig.suptitle("XLD-L2D Rig v12 — flip-free  |  depth-based parallax", fontsize=18)
for k, (name, yaw, pitch) in enumerate(POSES):
    ax = fig.add_subplot(2, 3, k+1)
    sx = np.sin(np.deg2rad(yaw)) * R_X
    sy = np.sin(np.deg2rad(pitch)) * R_Y
    x2 = xs + bx * sx
    y2 = ys + by * sy
    verts = np.stack([x2[tris], y2[tris]], axis=2)   # (n_t, 3, 2)
    # ★ Y 뒤집기 제거. 원본 픽셀 좌표(y 아래로 증가) 그대로 사용.
    pc = PolyCollection(verts, facecolors=tri_col, edgecolors='none')
    ax.add_collection(pc)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # ← Y축 반전 (위가 원점)
    ax.set_aspect('equal')
    ax.set_title(f"{name}  (yaw={yaw}°, pitch={pitch}°)", fontsize=12)
    ax.axis('off')
plt.tight_layout()
out1 = Path.home() / "XLD-L2D/assets/real/mesh/deform_rig_v12.png"
plt.savefig(out1, dpi=90, bbox_inches='tight')
print(f"[save] {out1}")

fig2, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=90)
for ax, basis, name in [(axes[0], bx, "basis_x"), (axes[1], by, "basis_y")]:
    verts = np.stack([xs[tris], ys[tris]], axis=2)
    norm = Normalize(vmin=-1, vmax=1)
    pc = PolyCollection(verts, array=basis[tris].mean(axis=1),
                        cmap='RdBu_r', norm=norm, edgecolors='none')
    ax.add_collection(pc)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # ★ 동일
    ax.set_aspect('equal')
    ax.set_title(name); ax.axis('off')
axes[2].imshow(src_np); axes[2].set_title("source"); axes[2].axis('off')
plt.tight_layout()
out2 = Path.home() / "XLD-L2D/assets/real/mesh/rig_v12_basis.png"
plt.savefig(out2, dpi=90, bbox_inches='tight')
print(f"[save] {out2}")
print("[done]")
