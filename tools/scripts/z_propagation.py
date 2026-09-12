import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import grey_dilation
import os

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
rgb = np.array(img).astype(np.float32)
H, W = rgb.shape[:2]
print("image %dx%d" % (W, H))

gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
gx = ndimage.sobel(gray, axis=1)
gy = ndimage.sobel(gray, axis=0)
gm = np.sqrt(gx*gx + gy*gy)
thr = np.percentile(gm, 94)
edge = gm > thr
print("edge: %d (%.2f%%)" % (edge.sum(), 100.0*edge.sum()/edge.size))

DIRS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
OPP = [(0,4),(1,5),(2,6),(3,7)]

def shift(a, dy, dx):
    out = np.zeros_like(a)
    yd = slice(max(0, dy), H + min(0, dy))
    xd = slice(max(0, dx), W + min(0, dx))
    ys = slice(max(0,-dy), H + min(0,-dy))
    xs = slice(max(0,-dx), W + min(0,-dx))
    out[yd, xd] = a[ys, xs]
    return out

MAXK = 6
MIN_STEM  = 2
MIN_CROSS = 4

streak = np.zeros((8, H, W), np.uint8)
for i, (dy, dx) in enumerate(DIRS):
    ok = np.ones((H, W), bool)
    for k in range(1, MAXK+1):
        ok = ok & shift(edge, k*dy, k*dx)
        streak[i] += ok.astype(np.uint8)

has_cross = np.zeros((H, W), bool)
for a, b in OPP:
    has_cross |= (streak[a] >= MIN_CROSS) & (streak[b] >= MIN_CROSS)
cand = edge & has_cross
print("candidates:", cand.sum())

stem_dir = np.full((H, W), -1, np.int8)
cross_idx = np.full((H, W), -1, np.int8)
strength = np.zeros((H, W), np.float32)

ys, xs = np.where(cand)
for y, x in zip(ys, xs):
    s = streak[:, y, x]
    best_pair_idx = -1; best_pair = None; best_min = 0
    for pi, (a, b) in enumerate(OPP):
        m = min(int(s[a]), int(s[b]))
        if m >= MIN_CROSS and m > best_min:
            best_min = m; best_pair = (a, b); best_pair_idx = pi
    if best_pair is None: continue
    stem_cands = []
    for i in range(8):
        if i in best_pair: continue
        si = int(s[i])
        if si >= MIN_STEM and si < best_min:
            stem_cands.append((i, si))
    if not stem_cands: continue
    stem_i, stem_s = max(stem_cands, key=lambda t: t[1])
    stem_dir[y, x] = stem_i
    cross_idx[y, x] = best_pair_idx
    strength[y, x] = stem_s + best_min

print("raw T:", (stem_dir >= 0).sum())

final = np.full((H, W), -1, np.int8)
final_cross = np.full((H, W), -1, np.int8)
RAD = 2
ys, xs = np.where(stem_dir >= 0)
order = np.argsort(-strength[ys, xs])
taken = np.zeros((H, W), bool)
for idx in order:
    y, x = int(ys[idx]), int(xs[idx])
    if taken[y, x]: continue
    y0, y1 = max(0,y-RAD), min(H,y+RAD+1)
    x0, x1 = max(0,x-RAD), min(W,x+RAD+1)
    if taken[y0:y1, x0:x1].any(): continue
    final[y, x] = stem_dir[y, x]
    final_cross[y, x] = cross_idx[y, x]
    taken[y0:y1, x0:x1] = True
print("after NMS:", (final >= 0).sum())

# ---- Z PROPAGATION ----
K_WALK = 15
votes_front = np.zeros((H, W), np.float32)
votes_back  = np.zeros((H, W), np.float32)

ys_f, xs_f = np.where(final >= 0)
print("walking %d T-junctions, K=%d" % (len(ys_f), K_WALK))

for y, x in zip(ys_f, xs_f):
    s  = int(final[y, x])
    cp = int(final_cross[y, x])
    if cp < 0: continue
    a, b = OPP[cp]
    dy, dx = DIRS[s]
    for k in range(1, K_WALK+1):
        yy, xx = y + k*dy, x + k*dx
        if yy < 0 or yy >= H or xx < 0 or xx >= W: break
        if not edge[yy, xx]: break
        votes_back[yy, xx] += 1
    for cd in (a, b):
        dy, dx = DIRS[cd]
        for k in range(1, K_WALK+1):
            yy, xx = y + k*dy, x + k*dx
            if yy < 0 or yy >= H or xx < 0 or xx >= W: break
            if not edge[yy, xx]: break
            votes_front[yy, xx] += 1

print("votes accumulated")
print("  front total: %.0f  back total: %.0f" % (votes_front.sum(), votes_back.sum()))
n_fo = int(((votes_front > 0) & (votes_back == 0)).sum())
n_bo = int(((votes_back > 0) & (votes_front == 0)).sum())
n_bt = int(((votes_front > 0) & (votes_back > 0)).sum())
n_nn = int((edge & (votes_front == 0) & (votes_back == 0)).sum())
print("  edge: front_only=%d back_only=%d both=%d neither=%d" % (n_fo, n_bo, n_bt, n_nn))

z = votes_front - votes_back
edge_vals = z[edge]
p2, p98 = np.percentile(edge_vals, [2, 98])
print("z 2/98pct: %.1f / %.1f" % (p2, p98))
z_norm = np.clip((z - p2) / (p98 - p2 + 1e-6), 0, 1)

# spread z to 3x3 for visibility
edge_wide = ndimage.binary_dilation(edge, iterations=1)
z_wide = grey_dilation(z_norm, size=(3,3))

r_ch = (z_wide * 255).astype(np.uint8)
g_ch = np.zeros_like(r_ch)
b_ch = ((1 - z_wide) * 255).astype(np.uint8)
edge_color = np.stack([r_ch, g_ch, b_ch], axis=-1)

vis = (rgb * 0.30 + 30).astype(np.uint8)
paint = edge_wide & (z_norm != np.median(z_norm))
vis[paint] = (0.9 * edge_color[paint] + 0.1 * vis[paint]).astype(np.uint8)

Image.fromarray(vis).save(f"{ROOT}/z_propagation.png")
print("saved z_propagation.png")

for name, box in [("hair",(200,300,1500,1000)),("face",(400,500,1000,1100))]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1, x0:x1]).save(f"{ROOT}/z_propagation_{name}.png")
print("saved crops")

hist, edges_ = np.histogram(edge_vals, bins=20)
print("\nz histogram:")
for i in range(20):
    bar = "#" * int(hist[i] / max(1, hist.max()) * 40)
    print("  %5.0f: %6d %s" % (edges_[i], hist[i], bar))
