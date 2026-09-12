#!/usr/bin/env python3
"""3way_fusion_v2.py — SCC-based topo layering (no relaxation drift)
A: depth prior only
B: topo only (T-junction)
C: blend(prior, topo)
"""
import sys
import numpy as np
from pathlib import Path
from collections import deque
from skimage import io, feature, morphology, color, transform
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

CANNY_SIGMA = 1.5
CANNY_LOW = 0.08
CANNY_HIGH = 0.20
N_BUCKETS = 7
MIN_REGION_SIZE = 800
WALK_LEN = 20

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
    bl = np.concatenate([lbl[0,:], lbl[-1,:], lbl[:,0], lbl[:,-1]])
    bl = bl[bl > 0]
    if len(bl) == 0: return np.ones((H, W), dtype=bool)
    bg_label = np.bincount(bl).argmax()
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
    edges = np.percentile(vals, np.linspace(0, 100, N_BUCKETS+1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    region_id = np.zeros(depth.shape, dtype=np.int32)
    nid = 1
    st = np.ones((3,3), dtype=np.uint8)
    for b in range(N_BUCKETS):
        m = char_mask & (d >= edges[b]) & (d < edges[b+1])
        lbl, n = ndimage.label(m, structure=st)
        for i in range(1, n+1):
            if int((lbl == i).sum()) < MIN_REGION_SIZE: continue
            region_id[lbl == i] = nid; nid += 1
    region_z = np.zeros(nid, dtype=np.float32)
    for rid in range(1, nid):
        m = region_id == rid
        if m.any(): region_z[rid] = d[m].mean()
    v = region_z[1:]
    if len(v) > 0 and v.max() > v.min():
        region_z[1:] = (v - v.min()) / (v.max() - v.min())
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
    cons = []
    for i in range(len(ys)):
        y, x = int(ys[i]), int(xs[i])
        neigh = []
        for dy, dx in DIRS:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                neigh.append((dy, dx))
        if len(neigh) != 3: continue
        best_dot = 1.0; cb = None
        for a in range(3):
            for b in range(a+1, 3):
                va = np.array(neigh[a], np.float32); va /= np.linalg.norm(va)
                vb = np.array(neigh[b], np.float32); vb /= np.linalg.norm(vb)
                d = float(np.dot(va, vb))
                if d < best_dot: best_dot = d; cb = (a, b)
        if cb is None: continue
        stem_i = [k for k in range(3) if k not in cb][0]
        segs = []
        for k in [stem_i, cb[0], cb[1]]:
            dy, dx = neigh[k]; ny, nx = y, x; hit = 0
            for _ in range(WALK_LEN):
                ny += dy; nx += dx
                if not (0 <= ny < H and 0 <= nx < W): break
                s = seg_lbl[ny, nx]
                if s > 0: hit = int(s); break
            segs.append(hit)
        if all(s > 0 for s in segs):
            cons.append(tuple(segs))
    return cons


def topo_layer(constraints, n_seg):
    """SCC-condensed topological layering. Returns z in [0,1] per segment."""
    u_list, v_list = [], []
    for s, ca, cb in constraints:
        u_list.append(ca); v_list.append(s)
        u_list.append(cb); v_list.append(s)
    if not u_list:
        return np.zeros(n_seg+1, np.float32), 1
    eu = np.array(u_list, dtype=np.int32)
    ev = np.array(v_list, dtype=np.int32)
    adj = csr_matrix((np.ones(len(eu), np.uint8), (eu, ev)),
                     shape=(n_seg+1, n_seg+1))
    n_scc, scc_lbl = connected_components(adj, directed=True, connection='strong')
    scc_u = scc_lbl[eu]; scc_v = scc_lbl[ev]
    cond = set((int(a), int(b)) for a, b in zip(scc_u, scc_v) if a != b)

    in_deg = np.zeros(n_scc, dtype=np.int32)
    out_adj = [[] for _ in range(n_scc)]
    for a, b in cond:
        in_deg[b] += 1; out_adj[a].append(b)

    q = deque([i for i in range(n_scc) if in_deg[i] == 0])
    topo_order = []
    while q:
        u = q.popleft(); topo_order.append(u)
        for v in out_adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0: q.append(v)

    if len(topo_order) < n_scc:
        print(f"[topo] WARN: {n_scc - len(topo_order)} SCCs unprocessed")
        for i in range(n_scc):
            if i not in topo_order: topo_order.append(i)

    layer_of_scc = np.zeros(n_scc, dtype=np.int32)
    for i, scc in enumerate(topo_order):
        layer_of_scc[scc] = i
    max_l = max(int(layer_of_scc.max()), 1)

    z = np.zeros(n_seg+1, dtype=np.float32)
    for s in range(1, n_seg+1):
        lyr = layer_of_scc[scc_lbl[s]]
        z[s] = 1.0 - lyr / max_l
    return z, n_scc


def colorize(z):
    z = np.clip(z, 0, 1)
    r = np.clip(1.5 - np.abs(z - 0.75)*4, 0, 1)
    g = np.clip(1.5 - np.abs(z - 0.50)*4, 0, 1)
    b = np.clip(1.5 - np.abs(z - 0.25)*4, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def render(name, sz, seg_lbl, cm, img, out):
    _, idx = ndimage.distance_transform_edt(seg_lbl == 0, return_indices=True)
    zmap = sz[seg_lbl[idx[0], idx[1]]]
    v = zmap[cm]
    if len(v) > 0 and v.max() > v.min():
        zmap = (zmap - v.min()) / (v.max() - v.min())
    vis = img[..., :3].copy()
    if vis.dtype != np.uint8:
        vis = np.clip(vis*255, 0, 255).astype(np.uint8)
    vis[cm] = colorize(zmap)[cm]
    io.imsave(Path(out) / f"{name}_zmap.png", vis)


def main(image_path, depth_path, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    img = io.imread(image_path); H, W = img.shape[:2]
    print(f"[load] {image_path}  {H}x{W}")

    cm = char_mask_from_border(img)
    print(f"[char] {int(cm.sum())} ({100*cm.mean():.1f}%)")

    depth = io.imread(depth_path)
    if depth.ndim == 3:
        depth = depth[..., 0] if depth.shape[-1] == 1 else depth.mean(-1)
    depth = depth.astype(np.float32)
    if depth.shape != (H, W):
        depth = transform.resize(depth, (H, W), anti_aliasing=True,
                                 preserve_range=True).astype(np.float32)
    print(f"[depth] {depth.min():.1f} .. {depth.max():.1f}")

    region_id, region_z = slice_regions(depth, cm)
    print(f"[regions] {int(region_id.max())}")

    gray = color.rgb2gray(img[..., :3]) if img.ndim == 3 else img.astype(np.float32)/255
    edges = feature.canny(gray, sigma=CANNY_SIGMA,
                          low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
    skel = morphology.skeletonize(edges)
    n_neigh = count_neighbors(skel.astype(np.uint8))
    t_mask = (n_neigh == 3) & skel
    ys, xs = np.where(t_mask)
    print(f"[tj] {len(ys)}")

    seg_lbl, n_seg = ndimage.label(skel & ~t_mask,
                                   structure=np.ones((3,3), np.uint8))
    print(f"[segments] {n_seg}")

    seg_z = compute_seg_z(seg_lbl, n_seg, region_id, region_z, skel)
    print(f"[seg_z] used {int((seg_z[1:] > 0).sum())}/{n_seg}")

    cons = find_t_constraints(skel, ys, xs, seg_lbl, H, W)
    print(f"[constraints] {len(cons)}")

    topo_z, n_scc = topo_layer(cons, n_seg)
    print(f"[scc] {n_scc}")

    # Prior violations
    n_pv = sum(1 for s, ca, cb in cons
               if seg_z[s] >= min(seg_z[ca], seg_z[cb]) - 0.01)
    print(f"[prior] T-violations {n_pv}/{len(cons)}")

    A_z = seg_z.copy()
    B_z = topo_z.copy()
    C_z = 0.5 * seg_z + 0.5 * topo_z

    for name, z in [('A', A_z), ('B', B_z), ('C', C_z)]:
        render(name, z, seg_lbl, cm, img, out)
        nz = z[1:]
        print(f"[{name}] z=[{nz.min():+.3f},{nz.max():+.3f}]  "
              f"n_unique={len(np.unique(np.round(nz,3)))}")

    d = np.abs(C_z[1:] - A_z[1:])
    print(f"[A vs C] mean={d.mean():.4f} max={d.max():.4f} >0.05:{int((d>0.05).sum())}")
    d2 = np.abs(B_z[1:] - A_z[1:])
    print(f"[A vs B] mean={d2.mean():.4f} max={d2.max():.4f}")

    print(f"[save] {out}/A_zmap.png B_zmap.png C_zmap.png")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "assets/real/my_vtuber.png"
    dep = sys.argv[2] if len(sys.argv) > 2 else "assets/real/depth_anything_v2.png"
    out = sys.argv[3] if len(sys.argv) > 3 else "assets/real/3way"
    main(img, dep, out)
