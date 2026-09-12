import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.morphology import skeletonize
from scipy import ndimage
import networkx as nx
import os
from collections import deque, Counter

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
rgb = np.array(img).astype(np.float32)
H, W = rgb.shape[:2]

gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
edge = canny(gray/255.0, sigma=1.5, low_threshold=0.08, high_threshold=0.20)
skel = skeletonize(edge)

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

t_mask = np.zeros((H, W), bool)
for (y, x, *_) in T_final:
    y0, y1 = max(0,y-2), min(H,y+3); x0, x1 = max(0,x-2), min(W,x+3)
    t_mask[y0:y1, x0:x1] = True

# --- 대신 T-junction을 "지우지 않고" 라벨에 포함 ---
# T-junction 픽셀만 lab=0으로 남기고, 나머지 skeleton flood fill
segment_skel = skel & ~branch     # branch 픽셀만 제외
lab, n_seg = ndimage.label(segment_skel, structure=np.ones((3,3), bool))
print("segments (branch-excluded): %d" % n_seg)

# 각 T-junction 위치: stem 방향 이웃의 segment, crossbar 두 방향 이웃의 segment
def get_neighbor_seg(y, x, dy, dx):
    yy, xx = y+dy, x+dx
    if 0 <= yy < H and 0 <= xx < W and lab[yy, xx] > 0:
        return int(lab[yy, xx])
    return 0

G = nx.DiGraph()
G.add_nodes_from(range(1, n_seg+1))
n_ok = 0
for (y, x, s_i, ca, cb) in T_final:
    ss = get_neighbor_seg(y, x, *DIRS[s_i])
    sa = get_neighbor_seg(y, x, *DIRS[ca])
    sb = get_neighbor_seg(y, x, *DIRS[cb])
    if ss > 0 and sa > 0 and sb > 0 and ss != sa and ss != sb:
        n_ok += 1
        G.add_edge(ss, sa)
        G.add_edge(ss, sb)

print("valid T constraints: %d" % n_ok)

# Condense + longest path layering
C = nx.condensation(G)
order_c = list(nx.topological_sort(C))
node_to_c = {}
for cid in range(len(C.nodes)):
    for m in C.nodes[cid]['members']:
        node_to_c[m] = cid

z_c = {}
for cid in order_c:
    preds = list(C.predecessors(cid))
    if not preds:
        z_c[cid] = 0
    else:
        z_c[cid] = max(z_c[p] for p in preds) + 1

z_seg = {n: z_c[node_to_c[n]] for n in range(1, n_seg+1)}
z_vals = list(z_seg.values())
print("z levels: min=%d max=%d" % (min(z_vals), max(z_vals)))
cnt = Counter(z_vals)
for lvl in sorted(cnt):
    print("  z=%2d: %d" % (lvl, cnt[lvl]))

# --- 새 시각화: z=0 = 회색(중립), z>=1 = 파랑→빨강 그라디언트 ---
zmax = max(z_vals) if z_vals else 1
z_img = np.full((H, W), -1.0, np.float32)
for s in range(1, n_seg+1):
    if z_seg[s] == 0:
        z_img[lab == s] = -1  # neutral marker
    else:
        z_img[lab == s] = z_seg[s] / zmax

vis = (rgb * 0.35 + 60).astype(np.uint8)

# neutral: 회색
neutral = (z_img == -1) & (lab > 0)
vis[neutral] = [120, 120, 120]

# constrained: gradient blue→red
const = (z_img > 0) & (lab > 0)
if const.any():
    zn = z_img[const]
    tint = np.stack([(zn*255).astype(np.uint8),
                     np.zeros_like(zn, np.uint8),
                     ((1-zn)*255).astype(np.uint8)], -1)
    vis[const] = tint

# branch 픽셀은 흰색
vis[branch] = [255, 255, 255]

Image.fromarray(vis).save(f"{ROOT}/z_segment_v4.png")
for name, box in [("hair",(200,300,1500,1000)),("face",(400,500,1000,1100))]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1, x0:x1]).save(f"{ROOT}/z_segment_v4_{name}.png")
print("saved z_segment_v4*.png")
