#!/usr/bin/env python3
"""3way_fusion.py — A/B/C comparison for fusion validation.
A: depth prior, no relaxation
B: uniform prior (0), with relaxation
C: depth prior, with relaxation (= fusion_v3)
"""
import sys
import numpy as np
from pathlib import Path
from skimage import io, feature, morphology, color, transform
from scipy import ndimage

CANNY_SIGMA = 1.5
CANNY_LOW = 0.08
CANNY_HIGH = 0.20
N_BUCKETS = 7
MIN_REGION_SIZE = 800
WALK_LEN = 20
RELAX_ITERS = 300
EPS = 0.01

DIRS = np.array([
    (-1,-1),(-1,0),(-1,1),
    (0,-1),(0,1),
    (1,-1),(1,0),(1,1),
], dtype=np.int32)


def char_mask_from_border(img, tol=40):
    H, W = img.shape[:2]
    border = np.concatenate([img[0,:], img[-1,:], img[:,0], img[:,-1]], axis=0)
    if border.ndim > 1:
        bg = np.median(border, axis=0)
        diff = np.linalg.norm(img.astype(np.float32) - bg, axis=-1)
    else:
        bg = np.median(border)
        diff = np.abs(img.astype(np.float32) - bg)
    similar = diff < tol
    lbl, n = ndimage.label(similar)
    if n == 0: return np.ones((H, W), dtype=bool)
    # pick label with most border pixels
    border_labels = np.concatenate([lbl[0,:], lbl[-1,:], lbl[:,0], lbl[:,-1]])
    border_labels = border_labels[border_labels > 0]
    if len(border_labels) == 0: return np.ones((H, W), dtype=bool)
    bg_label = np.bincount(border_labels).argmax()
    return ~(lbl == bg_label)


def count_neighbors(skel_u8):
    H, W = skel_u8.shape
    pad = np.pad(skel_u8, 1)
    n = np.zeros((H, W), dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1: continue
            n += pad[dy:dy+H, dx:dx+W]
    return n * skel_u8


def slice_regions(depth, char_mask):
    d = np.where(char_mask, depth, np.nan)
    vals = d[char_mask]
    if len(vals) == 0:
        return np.zeros_like(depth, np.int32), np.zeros(1, np.float32)
    qs = np.linspace(0, 100, N_BUCKETS+1)
    edges = np.percentile(vals, qs)
    edges[0] -= 1e-6; edges[-1] += 1e-6
    region_id = np.zeros(depth.shape, dtype=np.int32)
    next_id = 1
    st = np.ones((3,3), dtype=np.uint8)
    for b in range(N_BUCKETS):
        m = char_mask & (d >= edges[b]) & (d < edges[b+1])
        lbl, n = ndimage.label(m, structure=st)
        for i in range(1, n+1):
            if int((lbl == i).sum()) < MIN_REGION_SIZE: continue
            region_id[lbl == i] = next_id
            next_id += 1
    region_z = np.zeros(next_id, dtype=np.float32)
    for rid in range(1, next_id):
        m = region_id == rid
        if m.any(): region_z[rid] = d[m].mean()
    valid = region_z[1:]
    if len(valid) > 0 and valid.max() > valid.min():
        vmin, vmax = valid.min(), valid.max()
        region_z[1:] = (region_z[1:] - vmin) / (vmax - vmin)
    return region_id, region_z


def compute_seg_z(seg_lbl, n_seg, region_id, region_z, skel):
    _, idx = ndimage.distance_transform_edt(region_id == 0, return_indices=True)
    nearest_region = region_id[idx[0], idx[1]]
    skel_pix = skel & (seg_lbl > 0)
    seg_ids = seg_lbl[skel_pix]
    wts = region_z[nearest_region[skel_pix]]
    sums = np.bincount(seg_ids, weights=wts, minlength=n_seg+1)
    counts = np.bincount(seg_ids, minlength=n_seg+1)
    return sums / np.maximum(counts, 1)


def find_t_constraints(skel, ys, xs, seg_lbl, H, W):
    constraints = []
    for i in range(len(ys)):
        y, x = int(ys[i]), int(xs[i])
        neigh = []
        for dy, dx in DIRS:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                neigh.append((dy, dx))
        if len(neigh) != 3: continue
        best_dot = 1.0; cb_idx = None
        for a in range(3):
            for b in range(a+1, 3):
                va = np.array(neigh[a], dtype=np.float32); va /= np.linalg.norm(va)
                vb = np.array(neigh[b], dtype=np.float32); vb /= np.linalg.norm(vb)
                dot = float(np.dot(va, vb))
                if dot < best_dot: best_dot = dot; cb_idx = (a, b)
        if cb_idx is None: continue
        stem_i = [k for k in range(3) if k not in cb_idx][0]
        segs = []
        for k in [stem_i, cb_idx[0], cb_idx[1]]:
            dy, dx = neigh[k]; ny, nx = y, x; hit = 0
            for _ in range(WALK_LEN):
                ny += dy; nx += dx
                if not (0 <= ny < H and 0 <= nx < W): break
                s = seg_lbl[ny, nx]
                if s > 0: hit = int(s); break
            segs.append(hit)
        if all(s > 0 for s in segs):
            constraints.append(tuple(segs))
    return constraints


def relax(seg_z, constraints):
    sz = seg_z.copy(); n_viol = 0
    for it in range(RELAX_ITERS):
        viol = 0
        for s, ca, cb in constraints:
            target = min(sz[ca], sz[cb]) - EPS
            if sz[s] > target:
                sz[s] = target; viol += 1
        n_viol = viol
        if viol == 0: break
    return sz, n_viol


def colorize(z):
    z = np.clip(z, 0, 1)
    r = np.clip(1.5 - np.abs(z - 0.75)*4, 0, 1)
    g = np.clip(1.5 - np.abs(z - 0.50)*4, 0, 1)
    b = np.clip(1.5 - np.abs(z - 0.25)*4, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def render_and_save(name, sz_final, seg_lbl, char_mask, bg_img, out_dir):
    _, idx = ndimage.distance_transform_edt(seg_lbl == 0, return_indices=True)
    nearest_seg = seg_lbl[idx[0], idx[1]]
    zmap = sz_final[nearest_seg]
    valid = zmap[char_mask]
    if len(valid) > 0 and valid.max() > valid.min():
        zmap_n = (zmap - valid.min()) / (valid.max() - valid.min())
    else:
        zmap_n = np.zeros_like(zmap)
    vis = bg_img.copy()
    if vis.dtype != np.uint8:
        vis = np.clip(vis*255, 0, 255).astype(np.uint8)
    vis[char_mask] = colorize(zmap_n)[char_mask]
    io.imsave(Path(out_dir) / f"{name}_zmap.png", vis)
    return zmap_n


def main(image_path, depth_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    img = io.imread(image_path); H, W = img.shape[:2]
    print(f"[load] {image_path}  {H}x{W}")

    cm = char_mask_from_border(img)
    print(f"[char] pixels: {int(cm.sum())} ({100*cm.mean():.1f}%)")

    depth = io.imread(depth_path)
    if depth.ndim == 3:
        depth = depth[..., 0] if depth.shape[-1] == 1 else depth.mean(axis=-1)
    depth = depth.astype(np.float32)
    if depth.shape != (H, W):
        depth = transform.resize(depth, (H, W), anti_aliasing=True,
                                 preserve_range=True).astype(np.float32)
    print(f"[depth] range: {depth.min():.3f} .. {depth.max():.3f}")

    region_id, region_z = slice_regions(depth, cm)
    n_reg = int(region_id.max())
    print(f"[regions] n={n_reg}  z_range: {region_z[1:].min():.3f} .. {region_z[1:].max():.3f}")

    gray = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(np.float32)/255
    edges = feature.canny(gray, sigma=CANNY_SIGMA,
                          low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
    skel = morphology.skeletonize(edges)
    n_neigh = count_neighbors(skel.astype(np.uint8))
    t_mask = (n_neigh == 3) & skel
    ys, xs = np.where(t_mask)
    print(f"[tj] candidates: {len(ys)}")

    seg_lbl, n_seg = ndimage.label(skel & ~t_mask, structure=np.ones((3,3), dtype=np.uint8))
    print(f"[segments] n={n_seg}")

    seg_z = compute_seg_z(seg_lbl, n_seg, region_id, region_z, skel)
    print(f"[seg_z] assigned: {int((seg_z[1:] > 0).sum())}/{n_seg}")

    constraints = find_t_constraints(skel, ys, xs, seg_lbl, H, W)
    print(f"[constraints] n={len(constraints)}")
    print()

    print("=== 3-way comparison ===")
    results = {}
    for name, prior_seg_z in [('A', seg_z), ('B', None), ('C', seg_z)]:
        base = np.zeros_like(seg_z) if prior_seg_z is None else prior_seg_z
        if name == 'A':
            sz_final = base.copy(); n_viol = -1
        else:
            sz_final, n_viol = relax(base, constraints)
        render_and_save(name, sz_final, seg_lbl, cm, img, out)
        results[name] = sz_final
        nz = sz_final[1:]
        n_used = int((np.abs(nz) > 1e-6).sum())
        print(f"[{name}] viol={n_viol:>4}  "
              f"z=[{nz.min():+.3f}, {nz.max():+.3f}]  "
              f"n_unique={len(np.unique(np.round(nz, 3)))}  "
              f"n_used={n_used}/{n_seg}")

    diff = np.abs(results['C'][1:] - results['A'][1:])
    print(f"\n[A vs C] adjust mean={diff.mean():.4f}  max={diff.max():.4f}  "
          f">0.05: {int((diff > 0.05).sum())}/{len(diff)}")

    diff2 = np.abs(results['B'][1:] - results['A'][1:])
    print(f"[A vs B] diff   mean={diff2.mean():.4f}  max={diff2.max():.4f}")

    print(f"\n[save] {out}/A_zmap.png  B_zmap.png  C_zmap.png")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "assets/real/my_vtuber.png"
    dep = sys.argv[2] if len(sys.argv) > 2 else "assets/real/depth_anything_v2.png"
    out = sys.argv[3] if len(sys.argv) > 3 else "assets/real/3way"
    main(img, dep, out)
