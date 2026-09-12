import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.morphology import skeletonize
from scipy import ndimage
import os
from collections import deque

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
rgb = np.array(img).astype(np.float32)
H, W = rgb.shape[:2]

depth = np.array(Image.open(f"{ROOT}/depth_anything_v2.png").convert("L")).astype(np.float32)
print("depth loaded: %dx%d" % (W, H))

gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
edge = canny(gray/255.0, sigma=1.5, low_threshold=0.08, high_threshold=0.20)
skel = skeletonize(edge)
print("skeleton: %d px" % skel.sum())

DIRS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
DV = np.array(DIRS, dtype=np.float32)
skel_u8 = skel.astype(np.uint8)

def shift_arr(a, dy, dx):
    out = np.zeros_like(a)
    yd = slice(max(0,dy), H+min(0,dy)); xd = slice(max(0,dx), W+min(0,dx))
    ys_ = slice(max(0,-dy), H+min(0,-dy)); xs_ = slice(max(0,-dx), W+min(0,-dx))
    out[yd, xd] = a[ys_, xs_]
    return out

n_neigh = np.zeros((H, W), np.uint8)
for (dy, dx) in DIRS:
    n_neigh += shift_arr(skel_u8, dy, dx)
n_neigh[~skel] = 0

branch = skel & (n_neigh == 3)
ys, xs = np.where(branch)
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
    T_list.append((int(y), int(x), int(ndirs[stem_i]),
                   int(ndirs[best_pair[0]]), int(ndirs[best_pair[1]])))

used = np.zeros((H, W), bool)
T_final = []
for t in T_list:
    y, x = t[0], t[1]
    if used[y, x]: continue
    y0, y1 = max(0,y-3), min(H,y+4); x0, x1 = max(0,x-3), min(W,x+4)
    if used[y0:y1, x0:x1].any(): continue
    T_final.append(t)
    used[y0:y1, x0:x1] = True
print("T after NMS: %d" % len(T_final))

t_mask = np.zeros((H, W), bool)
for (y, x, *_) in T_final:
    y0, y1 = max(0,y-2), min(H,y+3); x0, x1 = max(0,x-2), min(W,x+3)
    t_mask[y0:y1, x0:x1] = True
segment_skel = skel & ~t_mask
lab, n_seg = ndimage.label(segment_skel, structure=np.ones((3,3), bool))
print("segments: %d" % n_seg)

def find_seg(y, x, dy, dx, max_walk=20):
    for k in range(1, max_walk+1):
        yy, xx = y+k*dy, x+k*dx
        if not (0 <= yy < H and 0 <= xx < W): break
        if lab[yy, xx] > 0: return int(lab[yy, xx])
    v0 = np.array([dy, dx], np.float32); v0 /= (np.linalg.norm(v0) + 1e-6)
    q = deque([(y+dy, x+dx, 1)]); visited = {(y+dy, x+dx)}
    while q:
        cy, cx, d = q.popleft()
        if d > max_walk: continue
        if lab[cy, cx] > 0: return int(lab[cy, cx])
        for ddy, ddx in DIRS:
            ny, nx = cy+ddy, cx+ddx
            if (ny, nx) in visited: continue
            if not (0 <= ny < H and 0 <= nx < W): continue
            vy = ny - y; vx = nx - x
            n = np.sqrt(vy*vy + vx*vx) + 1e-6
            if (vy*v0[0] + vx*v0[1]) / n < 0.5: continue
            visited.add((ny, nx))
            if skel[ny, nx] or lab[ny, nx] > 0:
                q.append((ny, nx, d+1))
    return 0

cstem = []; ccross = []
for (y, x, s_i, ca, cb) in T_final:
    ss = find_seg(y, x, *DIRS[s_i])
    sa = find_seg(y, x, *DIRS[ca])
    sb = find_seg(y, x, *DIRS[cb])
    if ss > 0 and sa > 0 and sb > 0 and ss != sa and ss != sb:
        cstem.append(ss); ccross.append(sa)
        cstem.append(ss); ccross.append(sb)
cstem = np.array(cstem, np.int32)
ccross = np.array(ccross, np.int32)
print("constraints: %d" % len(cstem))

# depth prior
sum_d = np.zeros(n_seg+1, np.float64)
cnt   = np.zeros(n_seg+1, np.int64)
ys_l, xs_l = np.where(lab > 0)
np.add.at(sum_d, lab[ys_l, xs_l], depth[ys_l, xs_l])
np.add.at(cnt,   lab[ys_l, xs_l], 1)
valid_seg = cnt > 0
z_prior = np.zeros(n_seg+1, np.float32)
dvals = sum_d[valid_seg] / cnt[valid_seg]
dmin, dmax = dvals.min(), dvals.max()
z_prior[valid_seg] = (dvals - dmin) / (dmax - dmin + 1e-6) * 100.0
print("prior: %.1f..%.1f  (n=%d)" % (
    z_prior[valid_seg].min(), z_prior[valid_seg].max(),
    int(valid_seg.sum())))

# vectorized relaxation
z = z_prior.copy().astype(np.float32)
MARGIN = 3.0
LAMBDA = 0.5
MU     = 0.02
ITER   = 300

for it in range(ITER):
    a = z[cstem]
    b = z[ccross]
    viol = a > (b - MARGIN)
    if not viol.any(): break
    # 적용할 delta 계산
    delta = (a - (b - MARGIN)) * viol
    # 누적
    np.add.at(z, cstem, -LAMBDA * 0.5 * delta)
    np.add.at(z, ccross, +LAMBDA * 0.5 * delta)
    z[valid_seg] += MU * (z_prior[valid_seg] - z[valid_seg])

print("iterations used: %d" % (it+1))

# check
viol = (z[cstem] > z[ccross]).sum()
print("remaining violations: %d / %d" % (viol, len(cstem)))

diff = z - z_prior
adj = np.abs(diff)
print("adjust: mean %.2f  max %.2f  >5: %d" % (
    adj[valid_seg].mean(), adj[valid_seg].max(),
    int((adj[valid_seg] > 5).sum())))

# visualize
zmin, zmax = z[valid_seg].min(), z[valid_seg].max()
z_norm = np.zeros_like(z)
z_norm[valid_seg] = (z[valid_seg] - zmin) / (zmax - zmin + 1e-6)

r_ch = (z_norm * 255).astype(np.uint8)
b_ch = ((1 - z_norm) * 255).astype(np.uint8)
tint = np.stack([r_ch, np.full_like(r_ch, 40), b_ch], -1)

vis = (rgb * 0.35 + 50).astype(np.uint8)
paint = lab > 0
seg_tint = tint[lab]
vis[paint] = (0.85 * seg_tint[paint] + 0.15 * vis[paint]).astype(np.uint8)
for (y, x, *_) in T_final:
    y0, y1 = max(0,y-1), min(H,y+2); x0, x1 = max(0,x-1), min(W,x+2)
    vis[y0:y1, x0:x1] = [255, 255, 255]

Image.fromarray(vis).save(f"{ROOT}/fusion_result.png")
for name, box in [("face",(400,500,1000,1100)),("hair",(200,300,1500,1000))]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1, x0:x1]).save(f"{ROOT}/fusion_result_{name}.png")
print("saved fusion_result*.png")
