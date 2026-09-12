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

# --- Kernel 1: edge (per-pixel) ---
gx = ndimage.sobel(gray, axis=1)
gy = ndimage.sobel(gray, axis=0)
gm = np.sqrt(gx*gx + gy*gy)
thr = np.percentile(gm, 95)   # top 5%
edge = gm > thr
print("edge: %d px (%.2f%%)" % (edge.sum(), 100.0*edge.sum()/edge.size))

# --- Kernel 2: 3x3 ring pattern ---
DIRS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
NAMES = ['E','NE','N','NW','W','SW','S','SE']

def shift(a, dy, dx):
    out = np.zeros_like(a)
    yd = slice(max(0, dy), H + min(0, dy))
    xd = slice(max(0, dx), W + min(0, dx))
    ys = slice(max(0,-dy), H + min(0,-dy))
    xs = slice(max(0,-dx), W + min(0,-dx))
    out[yd, xd] = a[ys, xs]
    return out

ring = np.zeros((8, H, W), dtype=bool)
for i,(dy,dx) in enumerate(DIRS):
    ring[i] = shift(edge, dy, dx)

# branch count = True→False 전이 수
ring_next = np.roll(ring, -1, axis=0)
n_branches = (ring & ~ring_next).sum(axis=0)
n_branches[~edge] = 0

cls = np.zeros((H,W), np.uint8)
cls[edge & (n_branches==1)] = 1   # endpoint
cls[edge & (n_branches==2)] = 2   # line or corner
cls[edge & (n_branches==3)] = 4   # T
cls[edge & (n_branches>=4)] = 5   # X

for k,n in [(1,'endpoint'),(2,'line/corner'),(4,'T'),(5,'X')]:
    print("  %-12s: %d" % (n, (cls==k).sum()))

# --- Kernel 3: T-stem 방향 (per-pixel, 7x7 streak) ---
MAXK = 3
cont = np.ones((8,H,W), dtype=bool)
for i,(dy,dx) in enumerate(DIRS):
    for k in range(1, MAXK+1):
        cont[i] &= shift(edge, k*dy, k*dx)

n_cont = cont.sum(axis=0)
t_strong = edge & (n_cont == 3)
print("T (streak>=3 in 3 dirs): %d" % t_strong.sum())

stem_dir = np.full((H,W), -1, np.int8)
t_ys, t_xs = np.where(t_strong)
DV = np.array(DIRS, dtype=np.float32)

for y,x in zip(t_ys, t_xs):
    dirs_here = [i for i in range(8) if cont[i,y,x]]
    if len(dirs_here) != 3:
        continue
    best_pair, best_dot = None, 2.0
    for a in range(3):
        for b in range(a+1,3):
            va = DV[dirs_here[a]]; vb = DV[dirs_here[b]]
            dot = float(np.dot(va, vb))
            if dot < best_dot:
                best_dot = dot; best_pair = (a,b)
    stem_idx = [i for i in range(3) if i not in best_pair][0]
    stem_dir[y,x] = dirs_here[stem_idx]

# --- 시각화 ---
vis = (rgb * 0.4 + 60).astype(np.uint8)
vis[edge] = [150,150,150]

palette = [
    (255, 60, 60),   # E
    (255,150, 60),   # NE
    (255,255, 60),   # N
    (150,255, 60),   # NW
    ( 60,255, 60),   # W
    ( 60,255,200),   # SW
    ( 60,160,255),   # S
    (150, 60,255),   # SE
]
for i in range(8):
    m = stem_dir == i
    n = m.sum()
    if n:
        print("  stem dir %s: %d px" % (NAMES[i], n))
        vis[m] = palette[i]

Image.fromarray(vis).save(f"{ROOT}/pixel_tjunction.png")
print("saved pixel_tjunction.png")

# 얼굴/머리카락 crop 확대 (T-junction 밀집 예상지역)
for name, box in [
    ("hair",  (200, 300, 1500, 1000)),
    ("face",  (400, 500, 1000, 1100)),
    ("body",  (600,1200, 1400,1800)),
]:
    x0,y0,x1,y1 = box
    crop = vis[y0:y1, x0:x1]
    Image.fromarray(crop).save(f"{ROOT}/pixel_tjunction_{name}.png")
    print("saved pixel_tjunction_%s.png" % name)
