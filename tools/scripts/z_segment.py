import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.morphology import skeletonize
from scipy import ndimage
import networkx as nx
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
print("T-candidates: %d" % len(ys))

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
print("valid T: %d" % len(T_list))

# NMS dist 3
used = np.zeros((H, W), bool)
T_final = []
for t in T_list:
    y, x = t[0], t[1]
    if used[y, x]: continue
    y0, y1 = max(0,y-3), min(H,y+4); x0, x1 = max(0,x-3), min(W,x+4)
    if used[y0:y1, x0:x1].any(): continue
    T_final.append(t)
    used[y0:y1, x0:x1] = True
print("after NMS: %d" % len(T_final))

# Remove T region to split skeleton
t_mask = np.zeros((H, W), bool)
for (y, x, *_) in T_final:
    y0, y1 = max(0,y-2), min(H,y+3)
    x0, x1 = max(0,x-2), min(W,x+3)
    t_mask[y0:y1, x0:x1] = True
segment_skel = skel & ~t_mask
print("segment skeleton px: %d" % segment_skel.sum())

structure = np.ones((3,3), dtype=bool)
lab, n_seg = ndimage.label(segment_skel, structure=structure)
print("segments: %d" % n_seg)

def find_segment(y, x, dy, dx, max_walk=15):
    for k in range(1, max_walk+1):
        yy, xx = y+k*dy, x+k*dx
        if not (0 <= yy < H and 0 <= xx < W): return 0
        if lab[yy, xx] > 0: return int(lab[yy, xx])
        if not skel[yy, xx]: return 0
    return 0

G = nx.DiGraph()
G.add_nodes_from(range(1, n_seg+1))
n_edges = 0
n_ok = 0
for (y, x, s_i, ca, cb) in T_final:
    seg_stem = find_segment(y, x, *DIRS[s_i])
    seg_a = find_segment(y, x, *DIRS[ca])
    seg_b = find_segment(y, x, *DIRS[cb])
    if seg_stem > 0 and seg_a > 0 and seg_b > 0 and seg_stem != seg_a and seg_stem != seg_b:
        n_ok += 1
        for sc in (seg_a, seg_b):
            if seg_stem != sc:
                G.add_edge(seg_stem, sc)
                n_edges += 1
print("T with 3 valid segs: %d" % n_ok)
print("constraint edges: %d" % n_edges)

# Topological sort with cycle handling
try:
    order = list(nx.topological_sort(G))
    z_seg = {node: i for i, node in enumerate(order)}
    print("topo sort OK")
except nx.NetworkXUnfeasible:
    C = nx.condensation(G)
    order_c = list(nx.topological_sort(C))
    z_seg = {}
    for i, cid in enumerate(order_c):
        for m in C.nodes[cid]['members']:
            z_seg[m] = i
    print("cycle detected, used condensation")

n_with_z = sum(1 for s in range(1, n_seg+1) if s in z_seg)
print("segments with z: %d / %d" % (n_with_z, n_seg))

if n_with_z == 0:
    print("no z assigned, aborting render")
    raise SystemExit

zmax = max(z_seg.values())
# Map to image
z_img = np.full((H, W), -1.0, np.float32)
for s in range(1, n_seg+1):
    if s in z_seg:
        z_img[lab == s] = z_seg[s] / zmax
print("zmax level: %d" % zmax)

# Visualize
vis = (rgb * 0.30 + 40).astype(np.uint8)
zmap = np.clip(z_img, 0, 1)
r_ch = (zmap * 255).astype(np.uint8)
g_ch = np.zeros_like(r_ch)
b_ch = ((1 - zmap) * 255).astype(np.uint8)
tint = np.stack([r_ch, g_ch, b_ch], axis=-1)
paint = lab > 0
vis[paint] = (0.85 * tint[paint] + 0.15 * vis[paint]).astype(np.uint8)

# T-junctions in white 3x3
for (y, x, *_) in T_final:
    y0, y1 = max(0, y-1), min(H, y+2)
    x0, x1 = max(0, x-1), min(W, x+2)
    vis[y0:y1, x0:x1] = [255, 255, 255]

Image.fromarray(vis).save(f"{ROOT}/z_segment.png")
print("saved z_segment.png")

for name, box in [("hair",(200,300,1500,1000)),("face",(400,500,1000,1100)),("body",(600,1200,1400,1800))]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1, x0:x1]).save(f"{ROOT}/z_segment_{name}.png")
print("saved crops")

# Histogram of z levels
zs_all = [z_seg[s] for s in range(1, n_seg+1) if s in z_seg]
hist, edges_ = np.histogram(zs_all, bins=min(20, max(2, zmax+1)))
print("\nz level histogram:")
for i in range(len(hist)):
    bar = "#" * int(hist[i] / max(1, hist.max()) * 40)
    print("  %2d: %5d %s" % (i, hist[i], bar))
