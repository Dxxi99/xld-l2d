#!/usr/bin/env python3
"""
tj_normalized.py — T-junction detection with normalized DV + deterministic tie-breaker
Step 1: normalizeDirections=true, tieBreaker=pixelIndexAsc (y*W+x)
"""
import sys
import numpy as np
from pathlib import Path
from skimage import io, feature, morphology, color

CANNY_SIGMA = 1.5
CANNY_LOW = 0.08
CANNY_HIGH = 0.20
NMS_RAD = 3
NORMALIZE = True

DIRS = np.array([
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
], dtype=np.int32)


def count_neighbors(skel_u8):
    H, W = skel_u8.shape
    pad = np.pad(skel_u8, 1, mode='constant')
    n = np.zeros((H, W), dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            n += pad[dy:dy+H, dx:dx+W]
    return n * skel_u8


def main(image_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = io.imread(image_path)
    if img.ndim == 3:
        gray = color.rgb2gray(img[..., :3])
    else:
        gray = img.astype(np.float32) / 255.0
    H, W = gray.shape
    print(f"[load] {image_path}  {H}x{W}")

    edges = feature.canny(gray, sigma=CANNY_SIGMA,
                          low_threshold=CANNY_LOW,
                          high_threshold=CANNY_HIGH)
    print(f"[canny] edge pixels: {int(edges.sum())} ({100*edges.mean():.2f}%)")

    skel = morphology.skeletonize(edges)
    print(f"[skeleton] pixels: {int(skel.sum())} ({100*skel.mean():.2f}%)")

    skel_u8 = skel.astype(np.uint8)
    n_neigh = count_neighbors(skel_u8)
    n_endpoint = int(((n_neigh == 1) & skel).sum())
    n_line     = int(((n_neigh == 2) & skel).sum())
    n_T        = int(((n_neigh == 3) & skel).sum())
    n_X        = int(((n_neigh == 4) & skel).sum())
    print(f"[neighbors] endpoint={n_endpoint}  line={n_line}  T/Y={n_T}  X={n_X}")

    ys, xs = np.where((n_neigh == 3) & skel)
    N = len(ys)
    strengths = np.zeros(N, dtype=np.float32)

    for i in range(N):
        y, x = int(ys[i]), int(xs[i])
        neigh_dirs = []
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                neigh_dirs.append((dy, dx))
        if len(neigh_dirs) != 3:
            continue
        v = np.array(neigh_dirs, dtype=np.float32)
        if NORMALIZE:
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
        pairs = [(0,1), (0,2), (1,2)]
        dots = [float(np.dot(v[a], v[b])) for a, b in pairs]
        best_dot = min(dots)
        strengths[i] = -best_dot

    s = strengths
    n_perfect   = int((s >  0.999).sum())
    n_imperfect = int(((s > 1e-6) & (s <= 0.999)).sum())
    n_zero      = int((s <= 1e-6).sum())
    print(f"[strength] total={N}")
    print(f"  perfect  (~1.0):    {n_perfect}")
    print(f"  imperfect (0<s<1):  {n_imperfect}")
    print(f"  zero     (~0):      {n_zero}")
    if N > 0:
        print(f"  min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}")

    pix_idx = ys.astype(np.int64) * W + xs.astype(np.int64)
    order = np.lexsort((pix_idx, -strengths))

    taken = np.zeros((H, W), dtype=bool)
    accepted_strength = np.full((H, W), -1.0, dtype=np.float32)
    accepted_list = []
    breakdown = {'strictly_weaker': 0, 'tie_broken': 0, 'accepted': 0}

    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        y0, y1 = max(0, y - NMS_RAD), min(H, y + NMS_RAD + 1)
        x0, x1 = max(0, x - NMS_RAD), min(W, x + NMS_RAD + 1)
        if taken[y0:y1, x0:x1].any():
            box = accepted_strength[y0:y1, x0:x1]
            mask = box >= 0
            if mask.any():
                blocker = float(box[mask].max())
                if blocker > strengths[idx]:
                    breakdown['strictly_weaker'] += 1
                else:
                    breakdown['tie_broken'] += 1
            continue
        taken[y0:y1, x0:x1] = True
        accepted_strength[y, x] = strengths[idx]
        accepted_list.append((y, x, float(strengths[idx])))
        breakdown['accepted'] += 1

    n_after = len(accepted_list)
    print(f"[NMS] RAD={NMS_RAD}")
    print(f"  accepted:           {breakdown['accepted']}")
    print(f"  rejected (weaker):  {breakdown['strictly_weaker']}")
    print(f"  rejected (tie):     {breakdown['tie_broken']}")
    print(f"  after NMS:          {n_after}")

    vis = img[..., :3].copy() if img.ndim == 3 else np.stack([img]*3, axis=-1)
    if vis.dtype != np.uint8:
        vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
    vis[skel] = (vis[skel] * 0.4 + np.array([0, 200, 200]) * 0.6).astype(np.uint8)
    vis[ys, xs] = [255, 255, 0]
    acc_y = np.array([p[0] for p in accepted_list]) if accepted_list else np.array([], dtype=int)
    acc_x = np.array([p[1] for p in accepted_list]) if accepted_list else np.array([], dtype=int)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            yy = np.clip(acc_y + dy, 0, H - 1)
            xx = np.clip(acc_x + dx, 0, W - 1)
            vis[yy, xx] = [255, 0, 0]

    vis_path = out / "tj_normalized_vis.png"
    io.imsave(vis_path, vis)
    print(f"[save] {vis_path}")

    npz_path = out / "tj_normalized_accept.npz"
    np.savez(npz_path, ys=acc_y, xs=acc_x,
             strengths=np.array([p[2] for p in accepted_list], dtype=np.float32))
    print(f"[save] {npz_path}")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "assets/real/my_vtuber.png"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "assets/real/tj_normalized"
    main(image_path, out_dir)
