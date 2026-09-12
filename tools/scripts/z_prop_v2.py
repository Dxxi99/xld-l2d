import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.morphology import skeletonize
import os

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
rgb = np.array(img).astype(np.float32)
H, W = rgb.shape[:2]
print("image %dx%d" % (W, H))

gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
edge = canny(gray/255.0, sigma=1.5, low_threshold=0.08, high_threshold=0.20)
skel = skeletonize(edge)
print("skeleton: %d px" % skel.sum())

DIRS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
DV = np.array(DIRS, dtype=np.float32)
NAMES = ['E','NE','N','NW','W','SW','S','SE']

# --- detect T/Y on skeleton ---
skel_u8 = skel.astype(np.uint8)
n_neigh = np.zeros((H, W), np.uint8)
for (dy, dx) in DIRS:
    s = np.zeros_like(skel_u8)
    yd = slice(max(0,dy), H+min(0,dy)); xd = slice(max(0,dx), W+min(0,dx))
    ys_ = slice(max(0,-dy), H+min(0,-dy)); xs_ = slice(max(0,-dx), W+min(0,-dx))
    s[yd, xd] = skel_u8[ys_, xs_]
    n_neigh += s
n_neigh[~skel] = 0

branch = skel & (n_neigh == 3)
ys, xs = np.where(branch)
print("T candidates: %d" % len(ys))

# each T-junction: record (stem_dir, crossbar_a, crossbar_b)
T_list = []
for y, x in zip(ys, xs):
    ndirs = []
    for i, (dy, dx) in enumerate(DIRS):
        yy, xx = y+dy, x+dx
        if 0 <= yy < H and 0 <= xx < W and skel[yy, xx]:
            ndirs.append(i)
    if len(ndirs) != 3: continue
    best_pair = None; best_dot = 2.0
    for a in range(3):
        for b in range(a+1,3):
            d = float(np.dot(DV[ndirs[a]], DV[ndirs[b]]))
            if d < best_dot: best_dot = d; best_pair = (a, b)
    stem_i = [i for i in range(3) if i not in best_pair][0]
    T_list.append((y, x, ndirs[stem_i], ndirs[best_pair[0]], ndirs[best_pair[1]]))

print("valid T: %d" % len(T_list))

# --- NMS by distance 3 ---
used = np.zeros((H, W), bool)
T_final = []
for (y, x, s_i, ca, cb) in T_list:
    if used[y, x]: continue
    y0, y1 = max(0,y-3), min(H,y+4); x0, x1 = max(0,x-3), min(W,x+4)
    if used[y0:y1, x0:x1].any(): continue
    T_final.append((y, x, s_i, ca, cb))
    used[y0:y1, x0:x1] = True
print("after NMS: %d" % len(T_final))

# --- z propagation ---
K = 20
votes_front = np.zeros((H, W), np.float32)
votes_back  = np.zeros((H, W), np.float32)

for (y, x, s_i, ca, cb) in T_final:
    # stem: weight 1.0, walk K
    dy, dx = DIRS[s_i]
    for k in range(1, K+1):
        yy, xx = y+k*dy, x+k*dx
        if not (0 <= yy < H and 0 <= xx < W): break
        if not skel[yy, xx]: break
        votes_back[yy, xx] += 1.0
    # crossbar: weight 0.5 each
    for cd in (ca, cb):
        dy, dx = DIRS[cd]
        for k in range(1, K+1):
            yy, xx = y+k*dy, x+k*dx
            if not (0 <= yy < H and 0 <= xx < W): break
            if not skel[yy, xx]: break
            votes_front[yy, xx] += 0.5

print("front total: %.0f  back total: %.0f" % (votes_front.sum(), votes_back.sum()))

z = votes_front - votes_back
# balance: 어떤 픽셀도 front/back 안 받으면 0
total = votes_front + votes_back
z_norm = np.where(total > 0, z / (total + 1e-6), 0.0)  # -1..+1

# --- stats on skeleton ---
skel_vals = z_norm[skel]
n_zero = int((np.abs(skel_vals) < 0.01).sum())
n_front = int((skel_vals > 0.1).sum())
n_back  = int((skel_vals < -0.1).sum())
print("on skeleton: front %d, back %d, neutral %d" % (n_front, n_back, n_zero))

# histogram of z on skeleton
hist, edges_ = np.histogram(skel_vals, bins=10, range=(-1, 1))
print("\nz histogram on skeleton (-1..+1):")
for i in range(10):
    bar = "#" * int(hist[i] / max(1, hist.max()) * 40)
    print("  %+.1f: %7d %s" % (edges_[i], hist[i], bar))

# --- visualization ---
vis = (rgb * 0.30 + 40).astype(np.uint8)

# color skeleton by z_norm: blue(-1) → gray(0) → red(+1)
zmap = (np.clip(z_norm, -1, 1) + 1) / 2.0    # 0..1
r_ch = (zmap * 255).astype(np.uint8)
g_ch = (np.zeros_like(r_ch) + 40)
b_ch = ((1 - zmap) * 255).astype(np.uint8)
tint = np.stack([r_ch, g_ch, b_ch], axis=-1)

# only paint skeleton pixels that got votes
paint = skel & (total > 0)
vis[paint] = tint[paint]

Image.fromarray(vis).save(f"{ROOT}/z_prop_v2.png")
print("saved z_prop_v2.png")

for name, box in [("hair",(200,300,1500,1000)),("face",(400,500,1000,1100)),("body",(600,1200,1400,1800))]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1, x0:x1]).save(f"{ROOT}/z_prop_v2_{name}.png")
print("saved crops")
