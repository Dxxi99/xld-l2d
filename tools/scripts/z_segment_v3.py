import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.morphology import skeletonize
from scipy import ndimage
import networkx as nx
import os
from collections import deque

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
rgb = np.array(img).astype(np.float32)
H, W = rgb.shape[:2]

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
    T_list.append((int(y), int(x), int(ndirs[stem_i]), int(ndirs[best_pair[0]]), int(ndirs[best_pair[1]])))

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

G = nx.DiGraph()
G.add_nodes_from(range(1, n_seg+1))
n_ok = 0
for (y, x, s_i, ca, cb) in T_final:
    seg_stem = find_seg(y, x, *DIRS[s_i])
    seg_a = find_seg(y, x, *DIRS[ca])
    seg_b = find_seg(y, x, *DIRS[cb])
    if seg_stem > 0 and seg_a > 0 and seg_b > 0 and seg_stem != seg_a and seg_stem != seg_b:
        n_ok += 1
        G.add_edge(seg_stem, seg_a)
        G.add_edge(seg_stem, seg_b)
print("T valid: %d" % n_ok)

# condense cycles
C = nx.condensation(G)
order_c = list(nx.topological_sort(C))
# map node -> cond id
node_to_c = {}
for cid, members in enumerate(C.nodes):
    for m in C.nodes[cid]['members']:
        node_to_c[m] = cid

# longest-path layering on condensation DAG
z_c = {}
for cid in order_c:
    preds = list(C.predecessors(cid))
    if not preds:
        z_c[cid] = 0
    else:
        z_c[cid] = max(z_c[p] for p in preds) + 1

z_seg = {n: z_c[node_to_c[n]] for n in range(1, n_seg+1)}

z_vals = list(z_seg.values())
print("z_seg: min=%d max=%d" % (min(z_vals), max(z_vals)))
from collections import Counter
cnt = Counter(z_vals)
print("z level distribution (top 10):")
for lvl in sorted(cnt)[:10]:
    print("  z=%2d: %d segments" % (lvl, cnt[lvl]))
print("  ... total levels: %d" % len(cnt))

zmax = max(z_vals) if z_vals else 1

z_img = np.full((H, W), -1.0, np.float32)
for s in range(1, n_seg+1):
    z_img[lab == s] = z_seg[s] / max(1, zmax)
print("coverage: %d px painted" % (z_img >= 0).sum())

vis = (rgb * 0.30 + 40).astype(np.uint8)
zmap = np.clip(z_img, 0, 1)
tint = np.stack([(zmap*255).astype(np.uint8),
                 np.zeros_like(zmap, np.uint8),
                 ((1-zmap)*255).astype(np.uint8)], -1)
paint = (lab > 0) & (z_img >= 0)
vis[paint] = (0.85 * tint[paint] + 0.15 * vis[paint]).astype(np.uint8)

Image.fromarray(vis).save(f"{ROOT}/z_segment_v3.png")
for name, box in [("hair",(200,300,1500,1000)),("face",(400,500,1000,1100)),("body",(600,1200,1400,1800))]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1, x0:x1]).save(f"{ROOT}/z_segment_v3_{name}.png")
print("saved z_segment_v3*.png")
