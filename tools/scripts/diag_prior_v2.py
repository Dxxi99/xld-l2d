#!/usr/bin/env python3
"""diag_prior_v2.py — filter malformed/trivial constraints, re-measure."""
import numpy as np
from skimage import io, feature, morphology, color, transform
from scipy import ndimage

N_BUCKETS=7; MIN_REGION_SIZE=800; WALK_LEN=20
CANNY_SIGMA=1.5; CANNY_LOW=0.08; CANNY_HIGH=0.20
DIRS = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], np.int32)

img = io.imread("assets/real/my_vtuber.png"); H,W = img.shape[:2]
d_raw = io.imread("assets/real/depth_anything_v2.png")
d = d_raw[...,0] if d_raw.ndim==3 else d_raw
d = transform.resize(d.astype(np.float32), (H,W), anti_aliasing=True,
                     preserve_range=True) if d.shape!=(H,W) else d.astype(np.float32)

# char mask
border = np.concatenate([img[0,:], img[-1,:], img[:,0], img[:,-1]], 0)
if border.ndim>1:
    bg = np.median(border,0); diff = np.linalg.norm(img.astype(np.float32)-bg, axis=-1)
else:
    bg = np.median(border); diff = np.abs(img.astype(np.float32)-bg)
similar = diff<40; lbl,_ = ndimage.label(similar)
bl = np.concatenate([lbl[0,:],lbl[-1,:],lbl[:,0],lbl[:,-1]]); bl=bl[bl>0]
cm = ~(lbl == np.bincount(bl).argmax())

# regions
dd = np.where(cm, d, np.nan)
edges = np.percentile(dd[cm], np.linspace(0,100,N_BUCKETS+1))
edges[0]-=1e-6; edges[-1]+=1e-6
region_id = np.zeros((H,W), np.int32); nid=1
st = np.ones((3,3), np.uint8)
for b in range(N_BUCKETS):
    m = cm & (dd>=edges[b]) & (dd<edges[b+1])
    L,k = ndimage.label(m, structure=st)
    for i in range(1,k+1):
        if int((L==i).sum())<MIN_REGION_SIZE: continue
        region_id[L==i]=nid; nid+=1
region_z = np.zeros(nid, np.float32)
for rid in range(1,nid):
    m = region_id==rid
    if m.any(): region_z[rid] = dd[m].mean()
v = region_z[1:]
if len(v) and v.max()>v.min(): region_z[1:] = (v-v.min())/(v.max()-v.min())

# skeleton/segments
gray = color.rgb2gray(img[...,:3]) if img.ndim==3 else img.astype(np.float32)/255
edges_i = feature.canny(gray, sigma=CANNY_SIGMA, low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
skel = morphology.skeletonize(edges_i)
pad = np.pad(skel.astype(np.uint8), 1)
nbr = np.zeros((H,W), np.uint8)
for dy in range(3):
    for dx in range(3):
        if dy==1 and dx==1: continue
        nbr += pad[dy:dy+H, dx:dx+W]
nbr *= skel.astype(np.uint8)
t_mask = (nbr==3) & skel
ys, xs = np.where(t_mask)
seg_lbl, n_seg = ndimage.label(skel & ~t_mask, structure=np.ones((3,3), np.uint8))

# seg_z (region-based)
_, idx = ndimage.distance_transform_edt(region_id==0, return_indices=True)
nearest_reg = region_id[idx[0], idx[1]]
skp = skel & (seg_lbl>0)
sids = seg_lbl[skp]; wts = region_z[nearest_reg[skp]]
seg_z = np.bincount(sids, weights=wts, minlength=n_seg+1) / np.maximum(np.bincount(sids, minlength=n_seg+1), 1)

# also seg_region (dominant region per segment)
seg_region = np.zeros(n_seg+1, np.int32)
for s in range(1, n_seg+1):
    m = (seg_lbl==s) & skp
    if m.any():
        seg_region[s] = np.bincount(nearest_reg[m]).argmax()

# constraints with validation
def find_cons():
    cons_all = []
    for i in range(len(ys)):
        y,x = int(ys[i]), int(xs[i])
        neigh = []
        for dy,dx in DIRS:
            ny,nx = y+dy, x+dx
            if 0<=ny<H and 0<=nx<W and skel[ny,nx]: neigh.append((dy,dx))
        if len(neigh)!=3: continue
        bd=1.0; cb=None
        for a in range(3):
            for b in range(a+1,3):
                va=np.array(neigh[a],np.float32); va/=np.linalg.norm(va)
                vb=np.array(neigh[b],np.float32); vb/=np.linalg.norm(vb)
                dt=float(np.dot(va,vb))
                if dt<bd: bd=dt; cb=(a,b)
        if cb is None: continue
        si = [k for k in range(3) if k not in cb][0]
        segs=[]
        for k in [si,cb[0],cb[1]]:
            dy,dx=neigh[k]; ny,nx=y,x; hit=0
            for _ in range(WALK_LEN):
                ny+=dy; nx+=dx
                if not (0<=ny<H and 0<=nx<W): break
                ss = seg_lbl[ny,nx]
                if ss>0: hit=int(ss); break
            segs.append(hit)
        if all(s>0 for s in segs): cons_all.append(tuple(segs))
    return cons_all

cons_all = find_cons()

# Classify
distinct = [c for c in cons_all if len(set(c))==3]
malformed = [c for c in cons_all if len(set(c))<3]

# Signal check: does the constraint have depth info? (stem region != cb regions)
def has_signal(c):
    s, ca, cb = c
    rs, rca, rcb = seg_region[s], seg_region[ca], seg_region[cb]
    return rs != rca or rs != rcb

signal = [c for c in distinct if has_signal(c)]
no_signal = [c for c in distinct if not has_signal(c)]

print(f"constraints total:       {len(cons_all)}")
print(f"  malformed (dup segs):  {len(malformed)}")
print(f"  distinct segs:         {len(distinct)}")
print(f"    with region signal:  {len(signal)}")
print(f"    no region signal:    {len(no_signal)}")
print()

def viol(sz, cons, direction='stem_less', eps=0.01):
    if direction=='stem_less':
        return sum(1 for s,ca,cb in cons if sz[s] > min(sz[ca],sz[cb]) + eps)
    else:
        return sum(1 for s,ca,cb in cons if sz[s] < max(sz[ca],sz[cb]) - eps)

print("=" * 60)
print("Violations on SIGNAL constraints only")
print("=" * 60)
for nm, cons in [('all distinct', distinct), ('signal only', signal)]:
    n = len(cons)
    if n == 0: continue
    vl = viol(seg_z, cons, 'stem_less')
    vg = viol(seg_z, cons, 'stem_greater')
    print(f"  [{nm:14s}] n={n:>4}  stem_less viol={vl:>4} ({100*vl/n:.1f}%)  stem_greater viol={vg:>4} ({100*vg/n:.1f}%)")

print()
print("=" * 60)
print("Sample SIGNAL constraints (up to 10)")
print("=" * 60)
for i, (s,ca,cb) in enumerate(signal[:10]):
    print(f"  T#{i}: stem={s}(z={seg_z[s]:.3f}, r={seg_region[s]})  "
          f"cb={ca}(z={seg_z[ca]:.3f},r={seg_region[ca]}) "
          f"{cb}(z={seg_z[cb]:.3f},r={seg_region[cb]})")

print()
print("=" * 60)
print("Sample NO-SIGNAL constraints (up to 5)")
print("=" * 60)
for i, (s,ca,cb) in enumerate(no_signal[:5]):
    print(f"  T#{i}: stem={s}(z={seg_z[s]:.3f}, r={seg_region[s]})  "
          f"cb={ca}(z={seg_z[ca]:.3f},r={seg_region[ca]}) "
          f"{cb}(z={seg_z[cb]:.3f},r={seg_region[cb]})")
