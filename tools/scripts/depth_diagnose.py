import numpy as np
from PIL import Image
import os

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img   = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
depth = Image.open(f"{ROOT}/depth_anything_v2.png").convert("L")

rgb = np.array(img).astype(np.float32)
d   = np.array(depth).astype(np.float32)
H, W = rgb.shape[:2]

R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
mx = np.maximum(np.maximum(R, G), B)
mn = np.minimum(np.minimum(R, G), B)
sat = (mx - mn) / (mx + 1e-6)
val = mx / 255.0

# ---------- A) FACE BOX 5구간 ----------
fx0, fy0, fx1, fy1 = 234, 539, 919, 1157
qs = np.percentile(d, [20, 40, 60, 80])

vis = rgb.copy()
box = vis[fy0:fy1, fx0:fx1]
db  = d[fy0:fy1, fx0:fx1]
colors = [(255,80,80), (255,180,80), (255,255,80), (80,255,120), (80,180,255)]
masks = [
    db <  qs[0],
    (db >= qs[0]) & (db < qs[1]),
    (db >= qs[1]) & (db < qs[2]),
    (db >= qs[2]) & (db < qs[3]),
    db >= qs[3],
]
for m, c in zip(masks, colors):
    box[m] = (0.5 * box[m] + 0.5 * np.array(c)).astype(np.float32)

vis[fy0:fy0+3, fx0:fx1] = [255,255,255]
vis[fy1-3:fy1, fx0:fx1] = [255,255,255]
vis[fy0:fy1, fx0:fx0+3] = [255,255,255]
vis[fy0:fy1, fx1-3:fx1] = [255,255,255]

Image.fromarray(vis.astype(np.uint8)).save(f"{ROOT}/diag_facebox.png")
print("saved diag_facebox.png")

# ---------- B) HAIR 3가지 ----------
m_a = (B > R + 20) & (sat > 0.2) & (val > 0.3)
m_b = (sat > 0.25) & (val < 0.85) & (val > 0.15)

skin_loose = (R > 180) & (G > 160) & (B > 150) & (R >= G) & (G >= B) & (R - B < 40) & (R - B > -10)
bg_ish = (R > 230) & (G > 230) & (B > 230)
gray = (0.299*R + 0.587*G + 0.114*B)
gx = np.zeros_like(gray); gy = np.zeros_like(gray)
gx[:,1:-1] = gray[:,2:] - gray[:,:-2]
gy[1:-1,:] = gray[2:,:] - gray[:-2,:]
gm = np.sqrt(gx*gx + gy*gy)
edge = gm > 30
m_c = (~skin_loose) & (~bg_ish) & (~edge) & (val < 0.9)

print("\n[hair masks]")
print("  A (blue only)      : %8d px (%.2f%%)" % (m_a.sum(), 100.0*m_a.sum()/m_a.size))
print("  B (sat+val)        : %8d px (%.2f%%)" % (m_b.sum(), 100.0*m_b.sum()/m_b.size))
print("  C (not skin/bg/edge): %8d px (%.2f%%)" % (m_c.sum(), 100.0*m_c.sum()/m_c.size))

def dim_out(rgbimg, mask, factor=0.25):
    v = rgbimg.copy()
    v[~mask] = v[~mask] * factor
    return v

visA = dim_out(rgb, m_a)
visB = dim_out(rgb, m_b)
visC = dim_out(rgb, m_c)

combo = np.concatenate([visA, visB, visC], axis=1).astype(np.uint8)
Image.fromarray(combo).save(f"{ROOT}/diag_hair3.png")
print("saved diag_hair3.png")

# hair C depth histogram
dc = d[m_c]
print("\n[hair C depth histogram]")
hist, edges = np.histogram(dc, bins=16, range=(0, 255))
for i in range(16):
    bar = "#" * int(hist[i] / max(1, hist.max()) * 40)
    print("  %3d-%3d: %7d %s" % (edges[i], edges[i+1], hist[i], bar))

# hair C top-42% 만
top_y = int(0.42 * H)
m_c_top = m_c.copy(); m_c_top[top_y:, :] = False
dc_top = d[m_c_top]
print("\n[hair C top-42%% depth histogram]  n=%d" % m_c_top.sum())
hist, edges = np.histogram(dc_top, bins=16, range=(0, 255))
for i in range(16):
    bar = "#" * int(hist[i] / max(1, hist.max()) * 40)
    print("  %3d-%3d: %7d %s" % (edges[i], edges[i+1], hist[i], bar))
