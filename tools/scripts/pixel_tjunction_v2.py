import numpy as np
from PIL import Image
from scipy import ndimage
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
print("edge: %d px (%.2f%%)" % (edge.sum(), 100.0*edge.sum()/edge.size))

DIRS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
NAMES = ['E','NE','N','NW','W','SW','S','SE']
OPP = [(0,4),(1,5),(2,6),(3,7)]   # E-W, NE-SW, N-S, NW-SE

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

# crossbar 후보 있는 픽셀만
has_cross = np.zeros((H, W), bool)
for a, b in OPP:
    has_cross |= (streak[a] >= MIN_CROSS) & (streak[b] >= MIN_CROSS)

cand = edge & has_cross
print("candidates:", cand.sum())

stem_dir = np.full((H, W), -1, np.int8)
n_valid = 0

ys, xs = np.where(cand)
for y, x in zip(ys, xs):
    s = streak[:, y, x]
    best_pair = None
    best_min = 0
    for a, b in OPP:
        m = min(int(s[a]), int(s[b]))
        if m >= MIN_CROSS and m > best_min:
            best_min = m
            best_pair = (a, b)
    if best_pair is None:
        continue
    stem_cands = []
    for i in range(8):
        if i in best_pair:
            continue
        si = int(s[i])
        if si >= MIN_STEM and si < best_min:
            stem_cands.append((i, si))
    if not stem_cands:
        continue
    stem_i, stem_s = max(stem_cands, key=lambda t: t[1])
    stem_dir[y, x] = stem_i
    n_valid += 1

print("valid T-junctions:", n_valid)
for i in range(8):
    n = (stem_dir == i).sum()
    if n:
        print("  stem %s: %d" % (NAMES[i], n))

# 시각화
vis = (rgb * 0.35 + 80).astype(np.uint8)
vis[edge] = [110, 110, 110]

palette = [
    (255, 60, 60), (255,150, 60), (255,255, 60), (150,255, 60),
    ( 60,255, 60), ( 60,255,200), ( 60,160,255), (150, 60,255),
]
for i in range(8):
    m = stem_dir == i
    if m.any():
        vis[m] = palette[i]

Image.fromarray(vis).save(f"{ROOT}/pixel_tjunction_v2.png")
print("saved pixel_tjunction_v2.png")

for name, box in [
    ("hair",  (200, 300, 1500, 1000)),
    ("face",  (400, 500, 1000, 1100)),
]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1, x0:x1]).save(f"{ROOT}/pixel_tjunction_v2_{name}.png")
    print("saved pixel_tjunction_v2_%s.png" % name)
