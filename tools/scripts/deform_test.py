#!/usr/bin/env python3
"""deform_test.py — move face layer (L3), verify mesh deforms correctly."""
import numpy as np
from pathlib import Path
from skimage import io
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

cache = Path("assets/real/swift_cache")
meta = open(cache/"meta.txt").read().split()
H, W = int(meta[1]), int(meta[3])
layer_id = np.fromfile(cache/"expected_layer_id.bin", dtype=np.uint8).reshape(H, W)
img = io.imread("assets/real/my_vtuber.png")

# Decimated grid per layer
def layer_verts(l, stride=10):
    mask = layer_id == l
    mask_sm = ndimage.binary_closing(mask, iterations=2)
    ys, xs = np.where(mask_sm)
    return xs[::stride], ys[::stride], mask_sm

q = [-1e-06, 0.1538, 0.2926, 0.4200, 0.4961, 0.6009, 0.8520, 1.000001]
layer_z = [0.5*(q[i]+q[i+1])*0.5 for i in range(7)]

# Deform: move L3 (face) by dx=+50, dy=-30
def shift_layer_layerwise(layer_id, l_target, dx, dy):
    """Simple: pixels of layer l_target shift; others stay."""
    new_layer = layer_id.copy()
    mask = layer_id == l_target
    ys, xs = np.where(mask)
    nx = np.clip(xs + dx, 0, W-1)
    ny = np.clip(ys + dy, 0, H-1)
    new_layer[mask] = 255  # clear
    # only place where destination is currently empty OR same layer (paint order: target last)
    new_layer[ny, nx] = l_target
    return new_layer

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
views = [
    ("original", layer_id),
    ("face +50,-30", shift_layer_layerwise(layer_id, 3, 50, -30)),
    ("hair -40,+20", shift_layer_layerwise(layer_id, 0, -40, 20)),
]
for ax, (title, lid) in zip(axes, views):
    # Overlay original image dim + layer
    for l in range(7):
        m = lid == l
        if m.sum() == 0: continue
        ax.contour(m, levels=[0.5], colors=[plt.cm.turbo(l/6)], linewidths=0.8)
    ax.set_title(title)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("assets/real/mesh/deform_test.png", dpi=90)
plt.close()
print("[save] assets/real/mesh/deform_test.png")
