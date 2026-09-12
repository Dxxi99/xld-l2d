#!/usr/bin/env python3
"""mesh_deform_v2.py — hard layer assignment + Laplacian-smoothed offset (vectorized)."""
import json
import numpy as np
from pathlib import Path
from skimage import io
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

cache = Path("assets/real/swift_cache")
meta = open(cache/"meta.txt").read().split()
H, W = int(meta[1]), int(meta[3])
layer_id = np.fromfile(cache/"expected_layer_id.bin", dtype=np.uint8).reshape(H, W)

q = [-1e-06, 0.1538, 0.2926, 0.4200, 0.4961, 0.6009, 0.8520, 1.000001]
layer_z = [0.5*(q[i]+q[i+1])*0.5 for i in range(7)]

# Sample vertices
char_mask = layer_id < 7
ys, xs = np.where(char_mask)
stride = 12
ys_s = ys[::stride]; xs_s = xs[::stride]
n_v = len(ys_s)
v_layer = layer_id[ys_s, xs_s].astype(np.int32)
print(f"[grid] {n_v} verts")

# Triangulation
tri = Triangulation(xs_s, ys_s)
triangles = tri.triangles
cx = np.clip(xs_s[triangles].mean(axis=1).astype(int), 0, W-1)
cy = np.clip(ys_s[triangles].mean(axis=1).astype(int), 0, H-1)
keep = char_mask[cy, cx]
triangles = triangles[keep]
print(f"[tri] {len(triangles)} triangles")

# --- Build sparse Laplacian adjacency (via triangle edges) ---
# Vectorized: gather all 3 edges per triangle, dedupe
e1 = np.stack([triangles[:,0], triangles[:,1]], 1)
e2 = np.stack([triangles[:,1], triangles[:,2]], 1)
e3 = np.stack([triangles[:,2], triangles[:,0]], 1)
edges = np.concatenate([e1, e2, e3], 0)
edges = np.sort(edges, axis=1)
edges = np.unique(edges, axis=0)
print(f"[adj] {len(edges)} unique edges")

n = n_v
rows = edges[:,0]
cols = edges[:,1]
data = np.ones(len(edges), np.float32)
A = csr_matrix((data, (rows, cols)), shape=(n, n))
A = A + A.T  # symmetric (both directions)
# degree (row sums)
deg = np.asarray(A.sum(axis=1)).ravel()
# Laplacian normalization for smoothing: off_new = (1-alpha)*off + alpha * (A @ off)/deg
deg_inv = 1.0 / np.maximum(deg, 1)
print(f"[laplacian] built")

# --- Layer offsets ---
def get_offsets(case):
    off = np.zeros((7, 2), np.float32)
    if case == 'nod':
        off[3] = [150, -200]
        off[1] = [40, -60]
        off[0] = [-80, -30]
    elif case == 'turn':
        off[3] = [200, 0]
        off[2] = [100, 0]
        off[1] = [-40, 0]
        off[0] = [-100, 0]
    return off

def smooth_offsets(v_layer, layer_offsets, A, deg_inv, n_iter=60, alpha=0.5):
    """Vectorized Laplacian smoothing: off = (1-alpha)*off + alpha * neighbor_mean."""
    off = layer_offsets[v_layer].copy().astype(np.float32)  # (n, 2)
    for it in range(n_iter):
        # A @ off = sum of neighbor offsets; multiply by deg_inv to get mean
        nb_sum = A @ off             # (n, 2)
        nb_mean = nb_sum * deg_inv[:, None]
        off = (1 - alpha) * off + alpha * nb_mean
    return off

# --- Render ---
def render(ax, xs_d, ys_d, v_layer, title):
    ax.tripcolor(xs_d, ys_d, triangles,
                 facecolors=v_layer[triangles].mean(axis=1),
                 cmap='turbo', vmin=0, vmax=6, edgecolors='none', alpha=0.9)
    ax.set_xlim(-200, W+200); ax.set_ylim(H+200, -200)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
render(axes[0], xs_s, ys_s, v_layer, "idle")

for ax, case, title in [
    (axes[1], 'nod',  'nod (smoothed)'),
    (axes[2], 'turn', 'turn (smoothed)'),
]:
    layer_off = get_offsets(case)
    t0 = __import__('time').time()
    off_field = smooth_offsets(v_layer, layer_off, A, deg_inv, n_iter=60, alpha=0.5)
    print(f"[{case}] smoothed in {__import__('time').time()-t0:.2f}s")
    xs_d = xs_s + off_field[:, 0]
    ys_d = ys_s + off_field[:, 1]
    render(ax, xs_d, ys_d, v_layer, title)
    mag = np.linalg.norm(off_field, axis=1)
    print(f"[{case}] offset mag: min={mag.min():.1f} max={mag.max():.1f} mean={mag.mean():.1f}")
    if (v_layer==3).any():
        m3 = mag[v_layer==3]
        print(f"[{case}] face(L3) mag: min={m3.min():.1f} max={m3.max():.1f} mean={m3.mean():.1f}")
    if (v_layer==0).any():
        m0 = mag[v_layer==0]
        print(f"[{case}] hair0(L0) mag: min={m0.min():.1f} max={m0.max():.1f} mean={m0.mean():.1f}")

plt.tight_layout()
out = Path("assets/real/mesh"); out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "deform_smoothed.png", dpi=90)
plt.close()
print(f"[save] {out/'deform_smoothed.png'}")
