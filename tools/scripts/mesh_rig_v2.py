#!/usr/bin/env python3
"""mesh_rig_v2.py — semantic group-based rig (Head/Body/Legs)."""
import json
import numpy as np
from pathlib import Path
from skimage import io
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

cache = Path("assets/real/swift_cache")
meta = open(cache/"meta.txt").read().split()
H, W = int(meta[1]), int(meta[3])
layer_id = np.fromfile(cache/"expected_layer_id.bin", dtype=np.uint8).reshape(H, W)

char_mask = layer_id < 7
ys, xs = np.where(char_mask)
stride = 12
ys_s = ys[::stride]; xs_s = xs[::stride]
n_v = len(ys_s)
v_layer = layer_id[ys_s, xs_s].astype(np.int32)
print(f"[grid] {n_v} verts")

tri = Triangulation(xs_s, ys_s)
triangles = tri.triangles
cx = np.clip(xs_s[triangles].mean(axis=1).astype(int), 0, W-1)
cy = np.clip(ys_s[triangles].mean(axis=1).astype(int), 0, H-1)
keep = char_mask[cy, cx]
triangles = triangles[keep]
print(f"[tri] {len(triangles)} triangles")

e1 = np.stack([triangles[:,0], triangles[:,1]], 1)
e2 = np.stack([triangles[:,1], triangles[:,2]], 1)
e3 = np.stack([triangles[:,2], triangles[:,0]], 1)
edges = np.sort(np.concatenate([e1, e2, e3], 0), axis=1)
edges = np.unique(edges, axis=0)
A = csr_matrix((np.ones(len(edges), np.float32), (edges[:,0], edges[:,1])),
               shape=(n_v, n_v))
A = A + A.T
deg_inv = 1.0 / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)

# ============ SEMANTIC GROUPING ============
# L0-L1: 뒷머리+중간머리 (head)
# L2: 앞머리/머리 아래 (head)
# L3: 얼굴 (head)
# L4: 상체/옷 (body, 0.2x)
# L5: 하체/다리 (body, 0.1x)
# L6: 신발 (fixed, 0)
GROUP = np.array([0, 0, 0, 0, 1, 1, 2])  # 0=head, 1=body, 2=legs
GROUP_WEIGHT = np.array([1.0, 1.0, 1.0, 1.0, 0.25, 0.10, 0.0])

# Head pivot = neck
neck = np.array([W * 0.5, H * 0.50], np.float32)
print(f"[pivot] neck = {neck}")

# Layer anchors (from previous run)
anchors = np.zeros((7, 2), np.float32)
for l in range(7):
    m = layer_id == l
    if m.sum() == 0:
        anchors[l] = [W/2, H/2]; continue
    yy, xx = np.where(m)
    anchors[l] = [xx.mean(), yy.mean()]

# Distance from each layer anchor to neck
head_dist = np.linalg.norm(anchors - neck, axis=1)
print(f"[dist to neck]")
for l in range(7):
    print(f"  L{l}: {head_dist[l]:.0f}px")

# ============ RIG ============
def compute_layer_offsets(yaw_deg, pitch_deg):
    """
    yaw: rotate head left/right around neck.
    - Face layer (L3): moves +x for +yaw (facing right)
    - Back hair (L0): moves -x for +yaw (rotates away)
    pitch: head nod up/down around neck (z-axis small rotation).
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    offs = np.zeros((7, 2), np.float32)
    for l in range(7):
        if GROUP_WEIGHT[l] == 0: continue
        # local direction: from neck to layer anchor
        dx = anchors[l, 0] - neck[0]
        dy = anchors[l, 1] - neck[1]
        r = np.sqrt(dx*dx + dy*dy) + 1e-6
        # yaw: rotate around neck in x-z plane. Approx = horizontal displacement ∝ dx.
        # Layers behind neck (dx small) move little; face layer (dx near 0) — need different proxy.
        # Better proxy: rotate by yaw on (dx, dy) around neck, but restricted to x.
        # Use: dx * cos(yaw) - r * sin(yaw) ... but simpler is:
        # layer-specific: face layer offset = yaw * head_radius
        # Use sign based on anchor position relative to neck
        # Face (L3) is at y~1436, close to neck (1452). Back hair (L0) is at y~1041, above neck.
        # In VTuber: +yaw = face turns right = face layer shifts right; back hair shifts left.
        # Sign convention: dx_sign depends on layer semantic
        # L0, L1 (back hair): -1  (shifts opposite to yaw)
        # L2, L3 (front hair, face): +1
        # L4, L5 (body): +0.3 (follows slightly)
        sign = np.array([-1, -1, +1, +1, +0.3, +0.2, 0])[l]
        head_radius = 400.0  # px, how much head rotates
        offs[l, 0] = sign * GROUP_WEIGHT[l] * head_radius * np.sin(yaw)
        # pitch: nod up/down. Layers above neck move up (dy negative for +pitch).
        # L0-L3 above/near neck: move up for +pitch (looking up)
        # Simple: dy_sign based on anchor y vs neck y
        if anchors[l, 1] < neck[1]:  # above neck
            dy_sign = +1  # up
        else:
            dy_sign = -0.2
        offs[l, 1] = dy_sign * GROUP_WEIGHT[l] * 250.0 * (1 - np.cos(pitch)) * 3
        # Approximation: for small pitch, (1-cos) is quadratic; use sin instead for linear feel
        offs[l, 1] = dy_sign * GROUP_WEIGHT[l] * 200.0 * np.sin(pitch)
    return offs

def smooth_offsets(off_v, A, deg_inv, n_iter=80, alpha=0.5):
    off = off_v.copy()
    for _ in range(n_iter):
        nb_mean = (A @ off) * deg_inv[:, None]
        off = (1 - alpha) * off + alpha * nb_mean
    return off

# ============ RENDER GRID ============
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
cases = [
    (0,   0,   "idle"),
    (-25, 0,   "yaw -25°"),
    (25,  0,   "yaw +25°"),
    (0,  -20,  "pitch -20° (up)"),
    (0,   20,  "pitch +20° (down)"),
    (30,  15,  "yaw+pitch"),
]
for ax, (yaw, pitch, title) in zip(axes.ravel(), cases):
    layer_off = compute_layer_offsets(yaw, pitch)
    off_v = layer_off[v_layer].astype(np.float32)
    if yaw != 0 or pitch != 0:
        off_v = smooth_offsets(off_v, A, deg_inv)
    xs_d = xs_s + off_v[:, 0]
    ys_d = ys_s + off_v[:, 1]
    ax.tripcolor(xs_d, ys_d, triangles,
                 facecolors=v_layer[triangles].mean(axis=1),
                 cmap='turbo', vmin=0, vmax=6, edgecolors='none', alpha=0.9)
    ax.set_xlim(-300, W+300); ax.set_ylim(H+300, -300)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)

plt.tight_layout()
out = Path("assets/real/mesh"); out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "deform_rig_v2.png", dpi=80)
plt.close()
print(f"[save] {out/'deform_rig_v2.png'}")

with open(out / "rig_config_v2.json", "w") as f:
    json.dump({
        'n_layers': 7,
        'group': GROUP.tolist(),
        'group_weight': GROUP_WEIGHT.tolist(),
        'sign': [-1, -1, 1, 1, 0.3, 0.2, 0],
        'neck': neck.tolist(),
        'anchors': anchors.tolist(),
        'head_radius': 400.0,
        'pitch_scale': 200.0,
        'smooth_iters': 80,
        'smooth_alpha': 0.5,
    }, f, indent=2)
print(f"[save] {out/'rig_config_v2.json'}")
