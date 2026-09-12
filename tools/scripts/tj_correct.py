import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.morphology import skeletonize
from scipy import ndimage
import os

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
rgb = np.array(img).astype(np.float32)
H, W = rgb.shape[:2]
print("image %dx%d" % (W, H))

gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]

# Canny
edge = canny(gray/255.0, sigma=1.5, low_threshold=0.08, high_threshold=0.20)
print("canny edge: %d (%.2f%%)" % (edge.sum(), 100.0*edge.sum()/edge.size))

# Skeletonize to 1px
skel = skeletonize(edge)
print("skeleton  : %d (%.2f%%)" % (skel.sum(), 100.0*skel.sum()/skel.size))

DIRS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
DV = np.array(DIRS, dtype=np.float32)
NAMES = ['E','NE','N','NW','W','SW','S','SE']

# neighbor count
skel_u8 = skel.astype(np.uint8)
n_neighbors = np.zeros((H, W), np.uint8)
for (dy, dx) in DIRS:
    shifted = np.zeros_like(skel_u8)
    yd = slice(max(0, dy), H + min(0, dy))
    xd = slice(max(0, dx), W + min(0, dx))
    ys_ = slice(max(0,-dy), H + min(0,-dy))
    xs_ = slice(max(0,-dx), W + min(0,-dx))
    shifted[yd, xd] = skel_u8[ys_, xs_]
    n_neighbors += shifted
n_neighbors[~skel] = 0

n_end = int((n_neighbors == 1).sum())
n_line = int((n_neighbors == 2).sum())
n_T = int((n_neighbors == 3).sum())
n_X = int((n_neighbors >= 4).sum())
print("endpoint: %d" % n_end)
print("line    : %d" % n_line)
print("T/Y     : %d" % n_T)
print("X       : %d" % n_X)

# --- classify branch points ---
branch_mask = skel & (n_neighbors == 3)
ys, xs = np.where(branch_mask)

stem_dir = np.full((H, W), -1, np.int8)
strength = np.zeros((H, W), np.float32)

for y, x in zip(ys, xs):
    ndirs = []
    for i, (dy, dx) in enumerate(DIRS):
        yy, xx = y+dy, x+dx
        if 0 <= yy < H and 0 <= xx < W and skel[yy, xx]:
            ndirs.append(i)
    if len(ndirs) != 3:
        continue
    best_pair = None
    best_dot = 2.0
    for a in range(3):
        for b in range(a+1,3):
            va = DV[ndirs[a]]; vb = DV[ndirs[b]]
            d = float(np.dot(va, vb))
            if d < best_dot:
                best_dot = d
                best_pair = (a, b)
    stem_i = [i for i in range(3) if i not in best_pair][0]
    stem_dir[y, x] = ndirs[stem_i]
    strength[y, x] = -best_dot

print("valid T with stem:", int((stem_dir >= 0).sum()))

# --- NMS ---
final = np.full((H, W), -1, np.int8)
RAD = 3
ys, xs = np.where(stem_dir >= 0)
order = np.argsort(-strength[ys, xs])
taken = np.zeros((H, W), bool)
for idx in order:
    y, x = int(ys[idx]), int(xs[idx])
    if taken[y, x]: continue
    y0, y1 = max(0, y-RAD), min(H, y+RAD+1)
    x0, x1 = max(0, x-RAD), min(W, x+RAD+1)
    if taken[y0:y1, x0:x1].any(): continue
    final[y, x] = stem_dir[y, x]
    taken[y0:y1, x0:x1] = True

n_final = int((final >= 0).sum())
print("after NMS:", n_final)
for i in range(8):
    n = int((final == i).sum())
    if n: print("  stem %s: %d" % (NAMES[i], n))

# --- visualize with big dots ---
vis = (rgb * 0.35 + 70).astype(np.uint8)
vis[skel] = [120, 120, 120]

palette = [
    (255, 60, 60), (255,150, 60), (255,255, 60), (150,255, 60),
    ( 60,255, 60), ( 60,255,200), ( 60,160,255), (150, 60,255),
]
ys, xs = np.where(final >= 0)
for y, x in zip(ys, xs):
    i = int(final[y, x])
    y0, y1 = max(0, y-2), min(H, y+3)
    x0, x1 = max(0, x-2), min(W, x+3)
    vis[y0:y1, x0:x1] = palette[i]

Image.fromarray(vis).save(f"{ROOT}/tj_correct.png")
print("saved tj_correct.png")

for name, box in [("hair",(200,300,1500,1000)),("face",(400,500,1000,1100)),("body",(600,1200,1400,1800))]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1, x0:x1]).save(f"{ROOT}/tj_correct_{name}.png")
print("saved crops")
