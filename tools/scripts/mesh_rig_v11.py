#!/usr/bin/env python3
"""v11: triangle-level clamp + EPS mask + tighter budget"""
import numpy as np
from pathlib import Path
import sys

SRC = Path.home() / "XLD-L2D/assets/real/mesh/rig_v8_sig120_R100.npz"
d = np.load(SRC)
xs = d['xs'].astype(np.float64); ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
bx = d['basis_x'].astype(np.float64).copy()
by = d['basis_y'].astype(np.float64).copy()
R_X = float(d['R_X']); R_Y = float(d['R_Y'])
H, W = int(d['H']), int(d['W'])
n_v, n_t = len(xs), len(tris)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
EPS    = 1e-5
MAX_YAW, MAX_PITCH = 30.0, 15.0
sx_max = R_X * np.sin(np.deg2rad(MAX_YAW))
sy_max = R_Y * np.sin(np.deg2rad(MAX_PITCH))
print(f"[cfg] BUDGET={BUDGET} sx_max={sx_max:.1f} sy_max={sy_max:.1f}")

e_dir = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
eu = np.unique(np.sort(e_dir, axis=1), axis=0)
u = eu[:,0]; v = eu[:,1]
el = np.linalg.norm(np.stack([xs[u]-xs[v], ys[u]-ys[v]],1), axis=1)
lim_x = el * BUDGET / sx_max
lim_y = el * BUDGET / sy_max

# ---- edge-level sequential clamp with EPS mask ----
def seq_clamp(b, u, v, lim, n_iter=500):
    rng = np.random.default_rng(0)
    for it in range(n_iter):
        d_ = b[u] - b[v]
        over = np.abs(d_) - lim
        mask = over > EPS
        n_viol = int(mask.sum())
        if n_viol == 0:
            print(f"  [converged] iter {it}")
            return b, 0
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        for k in idx:
            iu, iv = u[k], v[k]
            diff = b[iu] - b[iv]
            if abs(diff) <= lim[k] + EPS: continue
            excess = abs(diff) - lim[k]
            sgn = 1.0 if diff > 0 else -1.0
            b[iu] -= 0.5 * excess * sgn
            b[iv] += 0.5 * excess * sgn
        if it < 3 or it % 50 == 0:
            d_ = b[u] - b[v]; ov = np.abs(d_)-lim
            print(f"  iter {it}: viol={n_viol} max_over={ov.max():.6f}")
    return b, n_viol

print("[seq clamp x]"); bx, _ = seq_clamp(bx, u, v, lim_x)
print("[seq clamp y]"); by, _ = seq_clamp(by, u, v, lim_y)

# ---- triangle-level clamp (flip 직접 해소) ----
def tri_clamp(bx, by, sx, sy, n_iter=2000):
    rng = np.random.default_rng(1)
    for it in range(n_iter):
        x2 = xs + bx*sx; y2 = ys + by*sy
        A = (x2[tris[:,1]]-x2[tris[:,0]])*(y2[tris[:,2]]-y2[tris[:,0]]) \
          - (y2[tris[:,1]]-y2[tris[:,0]])*(x2[tris[:,2]]-x2[tris[:,0]])
        bad = A < 0
        n_bad = int(bad.sum())
        if n_bad == 0:
            print(f"  [converged] iter {it}")
            return bx, by, 0
        # 각 bad triangle마다: 가장 작은 basis 조정으로 A 양수화
        tidx = np.where(bad)[0]
        rng.shuffle(tidx)
        for t in tidx:
            a, b_, c = tris[t]
            pa = np.array([xs[a], ys[a]]); pb = np.array([xs[b_], ys[b_]]); pc = np.array([xs[c], ys[c]])
            # 현재 위치
            pa2 = pa + np.array([bx[a]*sx, by[a]*sy])
            pb2 = pb + np.array([bx[b_]*sx, by[b_]*sy])
            pc2 = pc + np.array([bx[c]*sx, by[c]*sy])
            A_cur = (pb2-pa2)[0]*(pc2-pa2)[1] - (pb2-pa2)[1]*(pc2-pa2)[0]
            if A_cur >= 0: continue
            # x-deform만 조정: A' = A_rest + (b_b-b_a)*sx*(y_c-y_a) - (b_c-b_a)*sx*(y_b-y_a)
            A_rest = (pb-pa)[0]*(pc-pa)[1] - (pb-pa)[1]*(pc-pa)[0]
            yCA = ys[c]-ys[a]; yBA = ys[b_]-ys[a]
            # 우리가 필요한 추가량: A' >= A_rest*0.5 (half-shrink 허용)
            deficit = A_rest*0.5 - A_cur
            # basis 조정: (Δb_b - Δb_a)*sx*yCA - (Δb_c - Δb_a)*sx*yBA = deficit
            # 분배: b_a에 몰아주기
            if abs(yCA - yBA) > 1e-6:
                db_a = -deficit / (sx * (yCA - yBA))
            else:
                db_a = 0
            db_a = np.clip(db_a, -0.3, 0.3)
            bx[a] += db_a
        if it < 5 or it % 100 == 0:
            print(f"  iter {it}: bad_tris={n_bad}")
    x2 = xs + bx*sx; y2 = ys + by*sy
    A = (x2[tris[:,1]]-x2[tris[:,0]])*(y2[tris[:,2]]-y2[tris[:,0]]) \
      - (y2[tris[:,1]]-y2[tris[:,0]])*(x2[tris[:,2]]-x2[tris[:,0]])
    n_bad = int((A<0).sum())
    print(f"  [end] {n_bad} bad triangles")
    return bx, by, n_bad

print("\n[tri clamp @ yaw=30, pitch=15 (worst case)]")
sx30 = np.sin(np.deg2rad(30))*R_X
sy15 = np.sin(np.deg2rad(15))*R_Y
bx, by, nb = tri_clamp(bx, by, sx30, sy15)

# ---- verify ----
def count_flips(x, y, t):
    p0 = np.stack([x[t[:,0]], y[t[:,0]]], 1); p1 = np.stack([x[t[:,1]], y[t[:,1]]], 1); p2 = np.stack([x[t[:,2]], y[t[:,2]]], 1)
    cr = (p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0])
    return int((cr < 0).sum())

print("\n[flip grid]")
print(f"{'yaw\\p':>6}", "".join(f"{p:>8}" for p in [-15,0,15]))
total = 0
for yaw in [-30,-20,-10,0,10,20,30]:
    row = []
    for pitch in [-15,0,15]:
        sx = np.sin(np.deg2rad(yaw))*R_X; sy = np.sin(np.deg2rad(pitch))*R_Y
        nf = count_flips(xs+bx*sx, ys+by*sy, tris)
        row.append(nf); total += nf
    print(f"{yaw:>6}", "".join(f"{v:>8}" for v in row))
print(f"[total flips] {total}")
print(f"[basis_x] range={bx.max()-bx.min():.4f}")
print(f"[basis_y] range={by.max()-by.min():.4f}")

out = Path.home() / f"XLD-L2D/assets/real/mesh/rig_v11_b{int(BUDGET*100)}.npz"
np.savez(out, xs=xs, ys=ys, triangles=tris,
         basis_x=bx.astype(np.float32), basis_y=by.astype(np.float32),
         H=H, W=W, R_X=R_X, R_Y=R_Y, BUDGET=BUDGET)
print(f"[save] {out}")
