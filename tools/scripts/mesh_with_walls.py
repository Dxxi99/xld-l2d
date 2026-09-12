#!/usr/bin/env python3
"""mesh_with_walls.py — mesh + side walls between adjacent layers."""
import json
import numpy as np
from pathlib import Path
from skimage import io, measure
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def trace_layer_contour(mask, min_area=200):
    padded = np.pad(mask, 1)
    contours = measure.find_contours(padded.astype(float), 0.5)
    out = []
    for c in contours:
        if len(c) < 4: continue
        c = c - 1.0
        y = c[:, 0]; x = c[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area >= min_area: out.append(c)
    return out

def simplify_contour(c, eps=1.5):
    from skimage.measure import approximate_polygon
    return approximate_polygon(c, tolerance=eps)

def mesh_from_mask(mask, z, stride=10):
    H, W = mask.shape
    ys, xs = np.where(mask)
    ys_s = ys[::stride]; xs_s = xs[::stride]
    if len(ys_s) < 3: return None
    tri = Triangulation(xs_s, ys_s)
    triangles = tri.triangles
    cx = np.clip(xs_s[triangles].mean(axis=1).astype(int), 0, W-1)
    cy = np.clip(ys_s[triangles].mean(axis=1).astype(int), 0, H-1)
    keep = mask[cy, cx]
    triangles = triangles[keep]
    if len(triangles) == 0: return None
    verts = np.stack([xs_s.astype(np.float32),
                      ys_s.astype(np.float32),
                      np.full(len(xs_s), z, np.float32)], axis=1)
    return {'verts': verts, 'tris': triangles.astype(np.int32), 'z': float(z)}

def build_side_walls(layer_below, layer_above, z_below, z_above, stride=4):
    """
    Connect boundary of `layer_above` to nearest boundary of `layer_below`.
    Returns verts, tris arrays.
    """
    def boundary(m):
        er = ndimage.binary_erosion(m, iterations=1)
        return m & ~er
    b_below = boundary(layer_below)
    b_above = boundary(layer_above)
    if not b_below.any() or not b_above.any():
        return None

    ys_a, xs_a = np.where(b_above)
    ys_b, xs_b = np.where(b_below)
    # sample stride
    ys_a = ys_a[::stride]; xs_a = xs_a[::stride]
    ys_b = ys_b[::stride]; xs_b = xs_b[::stride]
    if len(ys_a) < 2 or len(ys_b) < 2:
        return None

    # For each above-point, find nearest below-point
    pts_b = np.stack([xs_b, ys_b], axis=1).astype(np.float32)
    pts_a = np.stack([xs_a, ys_a], axis=1).astype(np.float32)
    tree = cKDTree(pts_b)
    dist, idx = tree.query(pts_a, k=1)
    # filter: only keep pairs within reasonable distance
    mask_ok = dist < 30
    pts_a = pts_a[mask_ok]; idx = idx[mask_ok]
    if len(pts_a) < 2:
        return None
    matched_b = pts_b[idx]

    # Build verts: [a_pts(z_above) ; b_pts(z_below)]
    n = len(pts_a)
    verts = np.concatenate([
        np.stack([pts_a[:,0], pts_a[:,1], np.full(n, z_above, np.float32)], axis=1),
        np.stack([matched_b[:,0], matched_b[:,1], np.full(n, z_below, np.float32)], axis=1),
    ], axis=0).astype(np.float32)

    # Build quad strip: for i in 0..n-1, connect i (top) i+1 (top) i+n (bottom) i+1+n
    # Order by x to make strip sensible
    order = np.argsort(pts_a[:,0])
    verts_reordered = np.concatenate([
        verts[order], verts[n + order]
    ], axis=0)
    tris = []
    for i in range(n - 1):
        # top strip
        a0 = i; a1 = i+1
        b0 = n + i; b1 = n + i + 1
        tris.append([a0, a1, b1])
        tris.append([a0, b1, b0])
    return {'verts': verts_reordered, 'tris': np.array(tris, np.int32),
            'z': (z_below + z_above) * 0.5}

def save_obj(path, all_meshes, names):
    with open(path, 'w') as f:
        f.write("# XLD-L2D layered mesh with walls\n")
        offset = 0
        for name, m in zip(names, all_meshes):
            if m is None: continue
            f.write(f"o {name}\n")
            for v in m['verts']:
                f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.4f}\n")
            for t in m['tris']:
                f.write(f"f {t[0]+1+offset} {t[1]+1+offset} {t[2]+1+offset}\n")
            offset += len(m['verts'])
    print(f"[obj] {path}  verts={offset}")

def save_preview(path, meshes, names, walls, H, W):
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, 7))
    ax = axes[0]
    for i, m in enumerate(meshes):
        if m is None: continue
        v = m['verts']; t = m['tris']
        ax.tripcolor(v[:,0], v[:,1], t, facecolors=np.full(len(t), i),
                     cmap='turbo', vmin=0, vmax=7, edgecolors='none', alpha=0.7)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect('equal'); ax.set_title('Top-down (x, y)')
    ax = axes[1]
    for i, m in enumerate(meshes):
        if m is None: continue
        ax.scatter(m['verts'][:,0], m['verts'][:,2], s=0.3, c=[colors[i]],
                   label=names[i], alpha=0.4)
    # overlay walls as gray connectors
    for wi, w in enumerate(walls):
        if w is None: continue
        v = w['verts']
        n = len(v) // 2
        # draw vertical connecting lines
        for i in range(0, n, 5):
            ax.plot([v[i,0], v[n+i,0]], [v[i,2], v[n+i,2]],
                    color='gray', alpha=0.15, linewidth=0.3)
    ax.set_xlabel('x'); ax.set_ylabel('z (depth)')
    ax.set_title('Side view (x, z) + wall connectors')
    ax.legend(markerscale=10, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"[preview] {path}")

def main():
    cache = Path("assets/real/swift_cache")
    meta = open(cache/"meta.txt").read().split()
    H = int(meta[1]); W = int(meta[3]); nSeg = int(meta[5])
    layer_id = np.fromfile(cache/"expected_layer_id.bin", dtype=np.uint8).reshape(H, W)

    q = [-1e-06, 0.1538, 0.2926, 0.4200, 0.4961, 0.6009, 0.8520, 1.000001]
    layer_z = [0.5*(q[i]+q[i+1]) * 0.5 for i in range(7)]

    print(f"[config] H={H} W={W}")
    for i, z in enumerate(layer_z):
        print(f"  L{i}: z={z:.3f}")

    stride = 10  # decimated from 6
    meshes = []
    names = []
    for l in range(7):
        mask = layer_id == l
        if mask.sum() == 0:
            meshes.append(None); names.append(f"L{l}"); continue
        mask_sm = ndimage.binary_closing(mask, iterations=2)
        m = mesh_from_mask(mask_sm, layer_z[l], stride=stride)
        meshes.append(m); names.append(f"L{l}")
        if m:
            print(f"  L{l}: verts={len(m['verts']):>6}  tris={len(m['tris']):>6}")

    # walls between adjacent layers
    walls = []
    wall_names = []
    for l in range(6):
        if meshes[l] is None or meshes[l+1] is None:
            walls.append(None); wall_names.append(f"W{l}_{l+1}"); continue
        w = build_side_walls(layer_id == l, layer_id == l+1,
                             layer_z[l], layer_z[l+1], stride=4)
        walls.append(w); wall_names.append(f"W{l}_{l+1}")
        if w:
            print(f"  Wall L{l}→L{l+1}: verts={len(w['verts']):>6}  tris={len(w['tris']):>6}")

    all_m = meshes + [w for w in walls if w is not None]
    all_n = names + [n for n, w in zip(wall_names, walls) if w is not None]
    out = Path("assets/real/mesh"); out.mkdir(parents=True, exist_ok=True)
    save_obj(out / "layered_walls.obj", all_m, all_n)
    save_preview(out / "preview_walls.png", meshes, names, walls, H, W)

if __name__ == "__main__":
    main()
