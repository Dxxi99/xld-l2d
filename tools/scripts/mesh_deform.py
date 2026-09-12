#!/usr/bin/env python3
"""mesh_deform.py — unified mesh with per-vertex layer weights + deformation."""
import json
import numpy as np
from pathlib import Path
from skimage import io
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# --- Load ---
cache = Path("assets/real/swift_cache")
meta = open(cache/"meta.txt").read().split()
H, W = int(meta[1]), int(meta[3])
layer_id = np.fromfile(cache/"expected_layer_id.bin", dtype=np.uint8).reshape(H, W)
img = io.imread("assets/real/my_vtuber.png")

q = [-1e-06, 0.1538, 0.2926, 0.4200, 0.4961, 0.6009, 0.8520, 1.000001]
layer_z = [0.5*(q[i]+q[i+1])*0.5 for i in range(7)]

print("[config] H=%d W=%d" % (H, W))

# --- Layer masks (smoothed) ---
layer_masks = []
for l in range(7):
    m = layer_id == l
    if m.sum() == 0:
        layer_masks.append(m)
    else:
        layer_masks.append(ndimage.binary_closing(m, iterations=2))

# --- Unified vertex grid ---
# Sample ALL char pixels at stride 10, one vertex per pixel
char_mask = layer_id < 7
ys, xs = np.where(char_mask)
ys_s = ys[::12]; xs_s = xs[::12]
n_v = len(ys_s)
print(f"[grid] vertices = {n_v}")

# Vertex layer (from layer_id)
v_layer = layer_id[ys_s, xs_s].astype(np.int32)

# --- Per-vertex layer weights via inverse distance to each layer mask ---
# Distance transform per layer, sampled at vertex positions
print("[weights] computing per-layer distance transforms...")
dist_per_layer = np.zeros((7, n_v), np.float32)
for l in range(7):
    if layer_masks[l].sum() == 0:
        dist_per_layer[l] = 1e6
        continue
    # distance to layer l
    dist = ndimage.distance_transform_edt(~layer_masks[l])
    dist_per_layer[l] = dist[ys_s, xs_s]
print("[weights] done")

# Softmax-like blend: w_l = exp(-d_l / tau) / sum
tau = 3.0  # pixels (sharp boundary)
weights = np.exp(-dist_per_layer / tau)
w_sum = weights.sum(axis=0, keepdims=True)
weights = weights / np.maximum(w_sum, 1e-6)

# Ensure hard assignment: dominant weight boosts to at least 0.8 for interior
max_w = weights.max(axis=0)
print(f"[weights] interior vertices (max_w>0.9): {int((max_w > 0.9).sum())}/{n_v}")
print(f"[weights] boundary vertices (max_w<0.6): {int((max_w < 0.6).sum())}/{n_v}")

# --- Triangulation ---
tri = Triangulation(xs_s, ys_s)
triangles = tri.triangles
# Keep triangles fully inside char mask
cx = np.clip(xs_s[triangles].mean(axis=1).astype(int), 0, W-1)
cy = np.clip(ys_s[triangles].mean(axis=1).astype(int), 0, H-1)
keep = char_mask[cy, cx]
triangles = triangles[keep]
print(f"[tri] triangles = {len(triangles)}")

# --- Deformation ---
def deform(xs_base, ys_base, weights, offsets):
    """
    offsets: (7, 2) array of (dx, dy) per layer
    returns: deformed xs, ys
    """
    dx = weights.T @ offsets[:, 0]
    dy = weights.T @ offsets[:, 1]
    return xs_base + dx, ys_base + dy

# --- Render function ---
def render(ax, xs_d, ys_d, title):
    # Color by dominant layer
    dom = weights.argmax(axis=0)
    tri_d = Triangulation(xs_d, ys_d, triangles)
    colors = plt.cm.turbo(dom / 6.0)
    ax.tripcolor(xs_d, ys_d, triangles, facecolors=dom[triangles].mean(axis=1),
                 cmap='turbo', vmin=0, vmax=6, edgecolors='none', alpha=0.85)
    ax.set_xlim(-100, W+100); ax.set_ylim(H+100, -100)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)

# --- Test cases ---
offsets_idle = np.zeros((7, 2), np.float32)
offsets_nod = np.zeros((7, 2), np.float32)
# Head nod: top layers (0=back hair, 1, 2) move up; face (3) moves down slightly
offsets_nod[3] = [150, -200]   # face moves up-right
offsets_nod[1] = [40, -60]
offsets_nod[0] = [-80, -30]  # back hair moves left-up

offsets_turn = np.zeros((7, 2), np.float32)
# Turn head: face shifts right, back hair shifts left
offsets_turn[3] = [200, 0]
offsets_turn[2] = [100, 0]
offsets_turn[1] = [-40, 0]
offsets_turn[0] = [-100, 0]

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
render(axes[0], xs_s, ys_s, "idle (base)")
xs_n, ys_n = deform(xs_s, ys_s, weights, offsets_nod)
render(axes[1], xs_n, ys_n, "nod (face up-right, hair left)")
xs_t, ys_t = deform(xs_s, ys_s, weights, offsets_turn)
render(axes[2], xs_t, ys_t, "turn (face right, hair left)")

plt.tight_layout()
out = Path("assets/real/mesh"); out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "deform_weights.png", dpi=90)
plt.close()
print(f"[save] {out/'deform_weights.png'}")

# --- Save vertex data for Swift ---
np.savez_compressed(
    out / "vertices.npz",
    xs=xs_s.astype(np.float32),
    ys=ys_s.astype(np.float32),
    z=np.array([layer_z[v_layer[i]] for i in range(n_v)], np.float32),
    weights=weights.astype(np.float32),
    v_layer=v_layer.astype(np.int32),
    triangles=triangles.astype(np.int32),
    H=np.int32(H), W=np.int32(W),
)
print(f"[save] {out/'vertices.npz'}  ({n_v} verts, {len(triangles)} tris)")

# --- Metadata for Swift ---
with open(out / "deform_meta.json", "w") as f:
    json.dump({
        'n_vertices': int(n_v),
        'n_triangles': int(len(triangles)),
        'n_layers': 7,
        'layer_z': layer_z,
        'tau': tau,
        'stride': 12,
    }, f, indent=2)
print(f"[save] {out/'deform_meta.json'}")
