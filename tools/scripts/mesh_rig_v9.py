#!/usr/bin/env python3
"""v9: v8 결과에서 hard clamp로 잔여 flip 제거"""
import numpy as np
from pathlib import Path

SRC = Path.home() / "XLD-L2D/assets/real/mesh/rig_v8_sig120_R100.npz"
d = np.load(SRC)
xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
bx = d['basis_x'].astype(np.float64)
by = d['basis_y'].astype(np.float64)
R_X = float(d['R_X']); R_Y = float(d['R_Y'])
H, W = int(d['H']), int(d['W'])
BUDGET = float(d['BUDGET']) if 'BUDGET' in d.files else 0.5
n_v, n_t = len(xs), len(tris)
print(f"[load] n_v={n_v} R_X={R_X} R_Y={R_Y} BUDGET={BUDGET}")

e_dir = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
eu = np.unique(np.sort(e_dir, axis=1), axis=0)
u = eu[:,0]; v = eu[:,1]
el = np.linalg.norm(np.stack([xs[u]-xs[v], ys[u]-ys[v]],1), axis=1)

MAX_YAW, MAX_PITCH = 30.0, 15.0
sx_max = R_X * np.sin(np.deg2rad(MAX_YAW))
sy_max = R_Y * np.sin(np.deg2rad(MAX_PITCH))
lim_x = el * BUDGET / sx_max
lim_y = el * BUDGET / sy_max

def hard_clamp(b, u, v, lim, n_iter=3000, damping=0.6):
    for it in range(n_iter):
        d_ = b[u] - b[v]
        over = np.abs(d_) - lim
        mask = over > 0
        n_viol = int(mask.sum())
        if n_viol == 0:
            print(f"  [converged] iter {it}")
            return b, 0
        sum_b = np.zeros(n_v); cnt = np.zeros(n_v)
        np.add.at(sum_b, u[mask], b[u[mask]])
        np.add.at(sum_b, v[mask], b[v[mask]])
        np.add.at(cnt,  u[mask], 1.0)
        np.add.at(cnt,  v[mask], 1.0)
        touched = cnt > 0
        new_vals = sum_b[touched] / cnt[touched]
        b[touched] = damping*new_vals + (1-damping)*b[touched]
        if it < 5 or it % 100 == 0:
            print(f"  iter {it}: viol={n_viol}  max_over={over.max():.6f}")
    d_ = b[u] - b[v]
    n_viol = int((np.abs(d_) > lim).sum())
    print(f"  [end] {n_viol} violations")
    return b, n_viol

print("[hard clamp x]")
bx, nvx = hard_clamp(bx, u, v, lim_x, damping=0.6)
print("[hard clamp y]")
by, nvy = hard_clamp(by, u, v, lim_y, damping=0.6)

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

out = Path.home() / "XLD-L2D/assets/real/mesh/rig_v9_flipfree.npz"
np.savez(out, xs=xs, ys=ys, triangles=tris,
         basis_x=bx.astype(np.float32), basis_y=by.astype(np.float32),
         H=H, W=W, R_X=R_X, R_Y=R_Y)
print(f"[save] {out}")
