#!/usr/bin/env python3
"""v10: 진짜 sequential Gauss-Seidel clamp — 위반 edge 하나씩 직접 해소"""
import numpy as np
from pathlib import Path

SRC = Path.home() / "XLD-L2D/assets/real/mesh/rig_v8_sig120_R100.npz"
d = np.load(SRC)
xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
bx = d['basis_x'].astype(np.float64).copy()
by = d['basis_y'].astype(np.float64).copy()
R_X = float(d['R_X']); R_Y = float(d['R_Y'])
H, W = int(d['H']), int(d['W'])
BUDGET = 0.5
n_v, n_t = len(xs), len(tris)
print(f"[load] n_v={n_v} R_X={R_X} R_Y={R_Y}")

e_dir = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
eu = np.unique(np.sort(e_dir, axis=1), axis=0)
u = eu[:,0]; v = eu[:,1]
el = np.linalg.norm(np.stack([xs[u]-xs[v], ys[u]-ys[v]],1), axis=1)

MAX_YAW, MAX_PITCH = 30.0, 15.0
sx_max = R_X * np.sin(np.deg2rad(MAX_YAW))
sy_max = R_Y * np.sin(np.deg2rad(MAX_PITCH))
lim_x = el * BUDGET / sx_max
lim_y = el * BUDGET / sy_max

def seq_clamp(b, u, v, lim, n_iter=2000, relax=1.0):
    """edge-by-edge Gauss-Seidel: 위반 edge만 순차적으로 직접 해소"""
    b = b.copy()
    rng = np.random.default_rng(0)
    for it in range(n_iter):
        d = b[u] - b[v]
        over = np.abs(d) - lim
        mask = over > 0
        n_viol = int(mask.sum())
        if n_viol == 0:
            print(f"  [converged] iter {it}")
            return b, 0
        idx = np.where(mask)[0]
        rng.shuffle(idx)     # 순서 섞어서 편향 방지
        # 순차 처리 (위반 edge만 ~수백 개라 Python loop OK)
        for k in idx:
            iu, iv = u[k], v[k]
            diff = b[iu] - b[iv]
            lk = lim[k]
            if abs(diff) <= lk:
                continue  # 이전 edge 처리로 이미 해소됨
            excess = abs(diff) - lk
            sgn = 1.0 if diff > 0 else -1.0
            b[iu] -= 0.5 * relax * excess * sgn
            b[iv] += 0.5 * relax * excess * sgn
        if it < 5 or it % 100 == 0:
            # 새 max_over
            d = b[u] - b[v]
            ov = np.abs(d) - lim
            print(f"  iter {it}: viol={n_viol}  max_over={ov.max():.6f}")
    d = b[u] - b[v]
    n_viol = int((np.abs(d) > lim).sum())
    print(f"  [end] {n_viol} violations")
    return b, n_viol

print("[seq clamp x]")
bx, nvx = seq_clamp(bx, u, v, lim_x, relax=1.0)
print("[seq clamp y]")
by, nvy = seq_clamp(by, u, v, lim_y, relax=1.0)

def count_flips(x, y, t):
    p0 = np.stack([x[t[:,0]], y[t[:,0]]], 1)
    p1 = np.stack([x[t[:,1]], y[t[:,1]]], 1)
    p2 = np.stack([x[t[:,2]], y[t[:,2]]], 1)
    cr = (p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0])
    return int((cr < 0).sum())

print("\n[flip grid]")
print(f"{'yaw\\p':>6}", "".join(f"{p:>8}" for p in [-15,0,15]))
total = 0
for yaw in [-30,-20,-10,0,10,20,30]:
    row = []
    for pitch in [-15,0,15]:
        sx = np.sin(np.deg2rad(yaw)) * R_X
        sy = np.sin(np.deg2rad(pitch)) * R_Y
        nf = count_flips(xs + bx*sx, ys + by*sy, tris)
        row.append(nf); total += nf
    print(f"{yaw:>6}", "".join(f"{v:>8}" for v in row))
print(f"[total flips] {total}")
print(f"[basis_x] range={bx.max()-bx.min():.4f}")
print(f"[basis_y] range={by.max()-by.min():.4f}")

out = Path.home() / "XLD-L2D/assets/real/mesh/rig_v10_flipfree.npz"
np.savez(out, xs=xs, ys=ys, triangles=tris,
         basis_x=bx.astype(np.float32), basis_y=by.astype(np.float32),
         H=H, W=W, R_X=R_X, R_Y=R_Y)
print(f"[save] {out}")
