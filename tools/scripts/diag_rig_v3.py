#!/usr/bin/env python3
"""rig_v3.npz 찢어짐 정량 진단"""
import numpy as np
from pathlib import Path

P = Path.home() / "XLD-L2D/assets/real/mesh/rig_v3.npz"
d = np.load(P)
print(f"[load] {P}")
for k in d.files:
    print(f"  {k}: shape={d[k].shape} dtype={d[k].dtype}")

xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
bx = d['basis_x'].astype(np.float64)
by = d['basis_y'].astype(np.float64)

n_v = len(xs)
n_t = len(tris)
print(f"\n[mesh] n_vert={n_v}  n_tri={n_t}")

# --- 1) Weld 여부 확인: 중복 좌표 개수 ---
from scipy.spatial import cKDTree
pts = np.stack([xs, ys], axis=1)
tree = cKDTree(pts)
pairs = tree.query_pairs(r=0.5, output_type='ndarray')
print(f"\n[weld check] eps=0.5px  near-duplicate pairs = {len(pairs)}")
if len(pairs) > 0:
    # union-find로 실제 그룹 개수
    parent = np.arange(n_v)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n_v)])
    uniq, counts = np.unique(roots, return_counts=True)
    merged = (counts > 1).sum()
    print(f"  weld groups (size>1): {merged}")
    print(f"  verts that would merge: {(counts[counts>1]).sum()} -> {merged} (save {counts[counts>1].sum()-merged})")

# --- 2) Flip 개수 (여러 yaw/pitch) ---
def count_flips(x, y, t):
    p0 = np.stack([x[t[:,0]], y[t[:,0]]], 1)
    p1 = np.stack([x[t[:,1]], y[t[:,1]]], 1)
    p2 = np.stack([x[t[:,2]], y[t[:,2]]], 1)
    a = (p1[:,0]-p0[:,0])*(p2[:,0]-p0[:,0])*0  # dummy
    cr = (p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0])
    return int((cr < 0).sum())

# rest pose flip (참고)
print(f"\n[flip @ rest] {count_flips(xs, ys, tris)} / {n_t}")

R_X, R_Y = 400.0, 250.0
print("\n[flip grid]  yaw\\pitch")
print(f"{'':>8}", "".join(f"{p:>8}" for p in [-15,0,15]))
total_bad = 0
for yaw in [-30,-20,-10,0,10,20,30]:
    row = []
    for pitch in [-15,0,15]:
        sx = np.sin(np.deg2rad(yaw)) * R_X
        sy = np.sin(np.deg2rad(pitch)) * R_Y
        dx = bx * sx
        dy = by * sy
        nf = count_flips(xs+dx, ys+dy, tris)
        row.append(nf)
        total_bad += nf
    print(f"{yaw:>8}", "".join(f"{v:>8}" for v in row))
print(f"\n[total flips over grid] {total_bad}")

# --- 3) basis_x / basis_y 요약 ---
print(f"\n[basis_x] min={bx.min():.4f} max={bx.max():.4f} mean={bx.mean():.4f}")
print(f"[basis_y] min={by.min():.4f} max={by.max():.4f} mean={by.mean():.4f}")

# --- 4) basis_x의 gradient 크기 (계단식이면 큼) ---
from scipy.sparse import coo_matrix
# triangle edge로 이웃 구성
edges = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
edges = np.vstack([edges, edges[:,::-1]])
G = coo_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(n_v,n_v)).tocsr()
G.data[:] = 1.0
deg = np.asarray(G.sum(1)).ravel()
deg[deg==0] = 1
lap_bx = (G @ bx) / deg - bx
print(f"\n[basis_x smoothness] mean|L(bx)| = {np.abs(lap_bx).mean():.6f}  max = {np.abs(lap_bx).max():.6f}")
