import numpy as np
from PIL import Image
import os

try:
    from skimage.feature import canny
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("need: pip install scikit-image")

from scipy import ndimage

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
rgb = np.array(img).astype(np.float32)
H, W = rgb.shape[:2]
print("image %dx%d" % (W, H))
gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]

def shift(a, dy, dx):
    out = np.zeros_like(a)
    yd = slice(max(0, dy), H + min(0, dy))
    xd = slice(max(0, dx), W + min(0, dx))
    ys = slice(max(0,-dy), H + min(0,-dy))
    xs = slice(max(0,-dx), W + min(0,-dx))
    out[yd, xd] = a[ys, xs]
    return out

DIRS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
OPP = [(0,4),(1,5),(2,6),(3,7)]

def analyze(edge, name, MAXK=6, MIN_STEM=2, MIN_CROSS=4, RAD=2):
    print("\n=== %s ===" % name)
    print("  edge: %d px (%.2f%%)" % (edge.sum(), 100.0*edge.sum()/edge.size))

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
    print("  candidates: %d" % cand.sum())

    stem_dir = np.full((H, W), -1, np.int8)
    strength = np.zeros((H, W), np.float32)
    ys, xs = np.where(cand)
    for y, x in zip(ys, xs):
        s = streak[:, y, x]
        bp = None; bm = 0
        for a, b in OPP:
            m = min(int(s[a]), int(s[b]))
            if m >= MIN_CROSS and m > bm:
                bm = m; bp = (a, b)
        if bp is None: continue
        sc = []
        for i in range(8):
            if i in bp: continue
            si = int(s[i])
            if si >= MIN_STEM and si < bm: sc.append((i, si))
        if not sc: continue
        si_, ss_ = max(sc, key=lambda t: t[1])
        stem_dir[y, x] = si_
        strength[y, x] = ss_ + bm

    n_raw = int((stem_dir >= 0).sum())
    print("  raw T: %d" % n_raw)

    final = np.full((H, W), -1, np.int8)
    ys, xs = np.where(stem_dir >= 0)
    order = np.argsort(-strength[ys, xs])
    taken = np.zeros((H, W), bool)
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if taken[y, x]: continue
        y0,y1 = max(0,y-RAD), min(H,y+RAD+1)
        x0,x1 = max(0,x-RAD), min(W,x+RAD+1)
        if taken[y0:y1,x0:x1].any(): continue
        final[y, x] = stem_dir[y, x]
        taken[y0:y1,x0:x1] = True

    n_final = int((final >= 0).sum())
    print("  after NMS: %d" % n_final)
    return final

# --- A) Sobel raw ---
gx = ndimage.sobel(gray, axis=1); gy = ndimage.sobel(gray, axis=0)
gm = np.sqrt(gx*gx + gy*gy)
sobel_thr = np.percentile(gm, 94)
edge_sobel = gm > sobel_thr
final_sobel = analyze(edge_sobel, "Sobel 94pct")

# --- B) Sobel + skeleton ---
edge_skel = skeletonize(edge_sobel)
final_skel = analyze(edge_skel, "Sobel + skeletonize")

if HAS_SKIMAGE:
    # --- C) Canny sigma 1.0 ---
    edge_c1 = canny(gray/255.0, sigma=1.0, low_threshold=0.08, high_threshold=0.20)
    final_c1 = analyze(edge_c1, "Canny s=1.0 lo=.08 hi=.20")

    # --- D) Canny sigma 1.5 ---
    edge_c2 = canny(gray/255.0, sigma=1.5, low_threshold=0.08, high_threshold=0.20)
    final_c2 = analyze(edge_c2, "Canny s=1.5")

    # --- E) Canny + skeleton ---
    edge_c3 = skeletonize(edge_c2)
    final_c3 = analyze(edge_c3, "Canny s=1.5 + skeletonize")
else:
    edge_c1 = edge_c2 = edge_c3 = None
    final_c1 = final_c2 = final_c3 = None

def render(final, edge_mask, fname):
    vis = (rgb * 0.30 + 80).astype(np.uint8)
    vis[edge_mask] = [130, 130, 130]
    palette = [
        (255, 60, 60), (255,150, 60), (255,255, 60), (150,255, 60),
        ( 60,255, 60), ( 60,255,200), ( 60,160,255), (150, 60,255),
    ]
    for i in range(8):
        m = final == i
        if m.any(): vis[m] = palette[i]
    Image.fromarray(vis).save(f"{ROOT}/{fname}")
    print("saved", fname)

render(final_skel, edge_skel, "edgecmp_sobel_skel.png")
if HAS_SKIMAGE:
    render(final_c2, edge_c2, "edgecmp_canny.png")
    render(final_c3, edge_c3, "edgecmp_canny_skel.png")

print("\ndone")
