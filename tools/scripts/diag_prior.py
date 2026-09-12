#!/usr/bin/env python3
"""diag_prior.py — 4-hypothesis diagnostic for 93% prior violations."""
import numpy as np
from pathlib import Path
from skimage import io, feature, morphology, color, transform
from scipy import ndimage

# --- same config as 3way ---
N_BUCKETS = 7; MIN_REGION_SIZE = 800; WALK_LEN = 20
CANNY_SIGMA = 1.5; CANNY_LOW = 0.08; CANNY_HIGH = 0.20
DIRS = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], np.int32)

img_path = "assets/real/my_vtuber.png"
dep_path = "assets/real/depth_anything_v2.png"

# ============ 0. depth orientation check ============
d_raw = io.imread(dep_path)
print("=" * 60)
print("0. DEPTH IMAGE ORIENTATION")
print("=" * 60)
print(f"shape={d_raw.shape}  dtype={d_raw.dtype}")
if d_raw.ndim == 3:
    print(f"R==G==B? {np.allclose(d_raw[...,0], d_raw[...,1])}")
    d = d_raw[..., 0] if d_raw.shape[-1]==1 else d_raw.mean(-1)
else:
    d = d_raw
H, W = d.shape
print(f"depth mean {d.mean():.1f}  range {d.min()}..{d.max()}")

# Known positions (from phase 12): shoe=front(high), face=mid, bg=far(low)
# We don't know exact pixels, but sample grid
print("\nVertical scan (row fraction → mean depth in that row):")
for frac in [0.05, 0.2, 0.4, 0.55, 0.7, 0.85, 0.95]:
    row = int(H * frac)
    print(f"  row {row:4d} ({frac:.2f}): mean={d[row].mean():7.2f}  "
          f"center={d[row, W//2]:7.2f}")

# ============ 1. Prepare pieces ============
img = io.imread(img_path)
H2, W2 = img.shape[:2]
d_res = transform.resize(d.astype(np.float32), (H2, W2),
                        anti_aliasing=True, preserve_range=True) if d.shape != (H2, W2) else d.astype(np.float32)
if d_res.shape != (H2, W2):
    print(f"[WARN] depth resize failed, {d.shape} -> {d_res.shape}")

# char mask
border = np.concatenate([img[0,:], img[-1,:], img[:,0], img[:,-1]], 0)
if border.ndim > 1:
    bg = np.median(border, 0)
    diff = np.linalg.norm(img.astype(np.float32) - bg, axis=-1)
else:
    bg = np.median(border); diff = np.abs(img.astype(np.float32) - bg)
similar = diff < 40
lbl, n = ndimage.label(similar)
bl = np.concatenate([lbl[0,:], lbl[-1,:], lbl[:,0], lbl[:,-1]]); bl = bl[bl>0]
cm = ~(lbl == np.bincount(bl).argmax())

# regions
dd = np.where(cm, d_res, np.nan)
edges = np.percentile(dd[cm], np.linspace(0, 100, N_BUCKETS+1))
edges[0]-=1e-6; edges[-1]+=1e-6
region_id = np.zeros((H2, W2), np.int32); nid = 1
st = np.ones((3,3), np.uint8)
for b in range(N_BUCKETS):
    m = cm & (dd >= edges[b]) & (dd < edges[b+1])
    L, k = ndimage.label(m, structure=st)
    for i in range(1, k+1):
        if int((L==i).sum()) < MIN_REGION_SIZE: continue
        region_id[L==i] = nid; nid += 1
region_z = np.zeros(nid, np.float32)
for rid in range(1, nid):
    m = region_id == rid
    if m.any(): region_z[rid] = dd[m].mean()
v = region_z[1:]
if len(v) and v.max() > v.min():
    region_z[1:] = (v - v.min()) / (v.max() - v.min())
print(f"\n[regions] {nid-1}")

# skeleton / segments
gray = color.rgb2gray(img[...,:3]) if img.ndim == 3 else img.astype(np.float32)/255
edges_i = feature.canny(gray, sigma=CANNY_SIGMA,
                        low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
skel = morphology.skeletonize(edges_i)
pad = np.pad(skel.astype(np.uint8), 1)
nbr = np.zeros((H2, W2), np.uint8)
for dy in range(3):
    for dx in range(3):
        if dy==1 and dx==1: continue
        nbr += pad[dy:dy+H2, dx:dx+W2]
nbr *= skel.astype(np.uint8)
t_mask = (nbr == 3) & skel
ys, xs = np.where(t_mask)
seg_lbl, n_seg = ndimage.label(skel & ~t_mask, structure=np.ones((3,3), np.uint8))
print(f"[segments] {n_seg}")

# ============ 2. seg_z THREE ways ============
# (A) current: nearest region
_, idx = ndimage.distance_transform_edt(region_id == 0, return_indices=True)
nearest_reg = region_id[idx[0], idx[1]]
skp = skel & (seg_lbl > 0)
sids = seg_lbl[skp]; wts = region_z[nearest_reg[skp]]
sz_A = np.bincount(sids, weights=wts, minlength=n_seg+1) / np.maximum(np.bincount(sids, minlength=n_seg+1), 1)

# (B) smoothed depth sampled at segment pixels
d_smooth = ndimage.gaussian_filter(d_res, 2.5)
sz_B = np.zeros(n_seg+1, np.float32)
for s in range(1, n_seg+1):
    m = seg_lbl == s
    if m.sum() > 0:
        sz_B[s] = np.median(d_smooth[m])
vv = sz_B[1:]
if vv.max() > vv.min(): sz_B[1:] = (vv - vv.min()) / (vv.max() - vv.min())

# (C) raw depth (no smooth) sampled at segment pixels
sz_C = np.zeros(n_seg+1, np.float32)
for s in range(1, n_seg+1):
    m = seg_lbl == s
    if m.sum() > 0:
        sz_C[s] = np.median(d_res[m])
vv = sz_C[1:]
if vv.max() > vv.min(): sz_C[1:] = (vv - vv.min()) / (vv.max() - vv.min())

# ============ 3. Constraints ============
def find_cons():
    cons = []
    for i in range(len(ys)):
        y, x = int(ys[i]), int(xs[i])
        neigh = []
        for dy, dx in DIRS:
            ny, nx = y+dy, x+dx
            if 0<=ny<H2 and 0<=nx<W2 and skel[ny,nx]: neigh.append((dy,dx))
        if len(neigh)!=3: continue
        bd = 1.0; cb = None
        for a in range(3):
            for b in range(a+1,3):
                va=np.array(neigh[a],np.float32); va/=np.linalg.norm(va)
                vb=np.array(neigh[b],np.float32); vb/=np.linalg.norm(vb)
                dt = float(np.dot(va,vb))
                if dt<bd: bd=dt; cb=(a,b)
        if cb is None: continue
        si = [k for k in range(3) if k not in cb][0]
        segs=[]
        for k in [si,cb[0],cb[1]]:
            dy,dx=neigh[k]; ny,nx=y,x; hit=0
            for _ in range(WALK_LEN):
                ny+=dy; nx+=dx
                if not (0<=ny<H2 and 0<=nx<W2): break
                ss = seg_lbl[ny,nx]
                if ss>0: hit=int(ss); break
            segs.append(hit)
        if all(s>0 for s in segs): cons.append(tuple(segs))
    return cons
cons = find_cons()
print(f"[constraints] {len(cons)}")

def count_viol(sz, direction='stem_less'):
    if direction == 'stem_less':
        return sum(1 for s,ca,cb in cons if sz[s] >= min(sz[ca],sz[cb]) - 0.01)
    else:
        return sum(1 for s,ca,cb in cons if sz[s] <= max(sz[ca],sz[cb]) + 0.01)

print()
print("=" * 60)
print("1. PRIOR VIOLATIONS (default: stem_z < crossbar_z)")
print("=" * 60)
for nm, sz in [('A(region)', sz_A), ('B(smooth)', sz_B), ('C(raw)', sz_C)]:
    nv = count_viol(sz, 'stem_less')
    print(f"  seg_z {nm:10s}:  viol {nv:>4}/{len(cons)}  ({100*nv/len(cons):.1f}%)")

print()
print("=" * 60)
print("2. FLIPPED DIRECTION (stem_z > crossbar_z)")
print("=" * 60)
for nm, sz in [('A(region)', sz_A), ('B(smooth)', sz_B), ('C(raw)', sz_C)]:
    nv = count_viol(sz, 'stem_greater')
    print(f"  seg_z {nm:10s}:  viol {nv:>4}/{len(cons)}  ({100*nv/len(cons):.1f}%)")

print()
print("=" * 60)
print("3. DEPTH INVERTED? (1 - depth, then seg_z)")
print("=" * 60)
d_inv = 255.0 - d_res
# redo seg_z from inverted depth
sz_A_inv = np.zeros(n_seg+1, np.float32)
for s in range(1, n_seg+1):
    m = seg_lbl == s
    if m.sum() > 0:
        sz_A_inv[s] = np.median(d_inv[m])
vv = sz_A_inv[1:]
if vv.max() > vv.min(): sz_A_inv[1:] = (vv - vv.min()) / (vv.max() - vv.min())
nv = count_viol(sz_A_inv, 'stem_less')
print(f"  inverted + stem_less:  viol {nv:>4}/{len(cons)}")
nv = count_viol(sz_A_inv, 'stem_greater')
print(f"  inverted + stem_great: viol {nv:>4}/{len(cons)}")

print()
print("=" * 60)
print("4. SAMPLE T-JUNCTIONS (first 5)")
print("=" * 60)
for i, (s, ca, cb) in enumerate(cons[:5]):
    print(f"  T#{i}: stem={s} (z_A={sz_A[s]:.3f})  "
          f"cb=({ca}(z={sz_A[ca]:.3f}), {cb}(z={sz_A[cb]:.3f}))")
    print(f"        nearest region depths: stem_reg={nearest_reg[skel & (seg_lbl==s)][:1]}, "
          f"cb_regs={nearest_reg[skel & (seg_lbl==ca)][:1]},{nearest_reg[skel & (seg_lbl==cb)][:1]}")
