#!/usr/bin/env python3
"""mesh_from_layers.py — Layer mask → 2.5D mesh (contour + triangulation + z-offset).

Output:
  - per-layer mesh: vertices (x,y,z), triangles, uv
  - combined OBJ + MTL for visual inspection
  - zprofile.json: per-layer z for later Swift port
"""
import json
import numpy as np
from pathlib import Path
from skimage import io, measure, color
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

H_W = None

def trace_layer_contour(mask, min_area=200):
    """Extract contours from binary mask. Returns list of (N,2) float arrays."""
    # pad to close boundary contours touching image edge
    padded = np.pad(mask, 1, mode='constant')
    contours = measure.find_contours(padded.astype(float), 0.5)
    out = []
    for c in contours:
        if len(c) < 4: continue
        # c is (row, col) → (y, x), shift back by padding
        c = c - 1.0
        # polygon area
        y = c[:, 0]; x = c[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area >= min_area:
            out.append(c)
    return out

def simplify_contour(c, eps=1.5):
    """Douglas-Peucker simplification."""
    from skimage.measure import approximate_polygon
    return approximate_polygon(c, tolerance=eps)

def mesh_from_contours(contours, z_value, img_H, img_W):
    """
    Build a triangle mesh from a set of closed contours.
    Strategy: use matplotlib.tri.Triangulation on all interior points + contour pts,
    then mask out triangles whose centroid is outside the polygon.
    Simple, robust enough for prototype.
    """
    if not contours: return None

    # 1) Rasterize: make binary of this layer for interior test
    layer_bin = np.zeros((img_H, img_W), dtype=bool)
    for c in contours:
        # scanline fill for small polys — use skimage
        from skimage.draw import polygon as draw_poly
        rr, cc = draw_poly(c[:, 0], c[:, 1], shape=(img_H, img_W))
        layer_bin[rr, cc] = True
    # ensure outside pixels are False
    # (draw_poly fills inside; boundaries may be loose but ok for prototype)

    # 2) Sample interior grid — use stride for mesh density
    stride = 6
    ys, xs = np.where(layer_bin)
    ys_s = ys[::stride]; xs_s = xs[::stride]
    if len(ys_s) < 3: return None

    # 3) Delaunay on (x, y)
    tri = Triangulation(xs_s, ys_s)
    triangles = tri.triangles

    # 4) Filter triangles whose centroid is outside layer_bin
    cx = xs_s[triangles].mean(axis=1).astype(int)
    cy = ys_s[triangles].mean(axis=1).astype(int)
    cx = np.clip(cx, 0, img_W - 1)
    cy = np.clip(cy, 0, img_H - 1)
    keep = layer_bin[cy, cx]
    triangles = triangles[keep]

    if len(triangles) == 0: return None

    # 5) Vertices with z
    verts = np.stack([xs_s.astype(np.float32),
                      ys_s.astype(np.float32),
                      np.full(len(xs_s), z_value, np.float32)], axis=1)

    return {
        'verts': verts,
        'tris': triangles.astype(np.int32),
        'z': float(z_value),
    }

def save_obj(path, meshes, layer_names):
    """Save combined OBJ with per-layer material groups."""
    with open(path, 'w') as f:
        f.write("# XLD-L2D layered mesh\n")
        offset = 0
        for name, m in zip(layer_names, meshes):
            if m is None: continue
            f.write(f"o {name}\n")
            for v in m['verts']:
                f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.4f}\n")
            for t in m['tris']:
                f.write(f"f {t[0]+1+offset} {t[1]+1+offset} {t[2]+1+offset}\n")
            offset += len(m['verts'])
    print(f"[obj] {path}  total_verts={offset}")

def save_preview(path, meshes, layer_names, img_W, img_H):
    """Top-down + side view preview."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, 7))

    # Top-down
    ax = axes[0]
    for i, (name, m) in enumerate(zip(layer_names, meshes)):
        if m is None: continue
        v = m['verts']; t = m['tris']
        ax.tripcolor(v[:,0], v[:,1], t, facecolors=np.full(len(t), i),
                     cmap='turbo', vmin=0, vmax=7, edgecolors='none', alpha=0.7)
    ax.set_xlim(0, img_W); ax.set_ylim(img_H, 0)
    ax.set_aspect('equal'); ax.set_title('Top-down (x, y)')

    # Side (x, z)
    ax = axes[1]
    for i, (name, m) in enumerate(zip(layer_names, meshes)):
        if m is None: continue
        ax.scatter(m['verts'][:,0], m['verts'][:,2], s=0.5, c=[colors[i]],
                   label=name, alpha=0.5)
    ax.set_xlabel('x'); ax.set_ylabel('z (depth)')
    ax.set_title('Side view (x, z)')
    ax.legend(markerscale=10, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"[preview] {path}")

def main():
    # Load layer data from previous step
    cache = Path("assets/real/swift_cache")
    H = int(open(cache/"meta.txt").readline().split()[1])
    W = int(open(cache/"meta.txt").readlines()[1].split()[1])
    nSeg = int(open(cache/"meta.txt").readlines()[2].split()[1])

    layer_id = np.fromfile(cache/"expected_layer_id.bin", dtype=np.uint8).reshape(H, W)

    # Get per-layer z from quantile edges (mid of bucket)
    q = [-1e-06, 0.1538, 0.2926, 0.4200, 0.4961, 0.6009, 0.8520, 1.000001]
    layer_z = [0.5*(q[i]+q[i+1]) for i in range(7)]

    # Additional z-offset for visual separation (2.5D inflation)
    # Make depth span larger so mesh is visibly 3D
    z_span = 0.5  # total depth span in normalized units
    layer_z_3d = [lz * z_span for lz in layer_z]

    print(f"[config] H={H} W={W}")
    print(f"[config] layer_z: {[f'{z:.3f}' for z in layer_z_3d]}")

    meshes = []
    layer_names = []
    for l in range(7):
        mask = layer_id == l
        n = int(mask.sum())
        if n == 0:
            meshes.append(None); layer_names.append(f"L{l}")
            continue
        # smooth mask slightly for cleaner contour
        mask_smooth = ndimage.binary_closing(mask, iterations=2)
        contours = trace_layer_contour(mask_smooth, min_area=200)
        # simplify
        contours = [simplify_contour(c, eps=1.5) for c in contours]
        m = mesh_from_contours(contours, layer_z_3d[l], H, W)
        meshes.append(m)
        layer_names.append(f"L{l}")
        if m is not None:
            print(f"  L{l}: mask={n:>7}px  contours={len(contours):>3}  "
                  f"verts={len(m['verts']):>6}  tris={len(m['tris']):>6}  z={layer_z_3d[l]:.3f}")
        else:
            print(f"  L{l}: mask={n:>7}px  contours={len(contours):>3}  mesh=FAIL")

    out = Path("assets/real/mesh"); out.mkdir(parents=True, exist_ok=True)
    save_obj(out / "layered.obj", meshes, layer_names)
    save_preview(out / "preview.png", meshes, layer_names, W, H)

    # zprofile for Swift port
    with open(out / "zprofile.json", "w") as f:
        json.dump({
            'H': H, 'W': W,
            'n_layers': 7,
            'layer_z': layer_z_3d,
            'quantile_edges': q,
        }, f, indent=2)
    print(f"[save] {out}/layered.obj  preview.png  zprofile.json")

if __name__ == "__main__":
    main()
