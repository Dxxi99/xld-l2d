#!/usr/bin/env python3
"""cache_for_swift.py — dump all inputs for Swift LayerPromotion validation."""
import numpy as np
from pathlib import Path
from skimage import io, feature, morphology, color, transform
from scipy import ndimage

N_BUCKETS=7; MIN_REGION_SIZE=800; WALK_LEN=20
CANNY_SIGMA=1.5; CANNY_LOW=0.08; CANNY_HIGH=0.20
DIRS = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], np.int32)

img = io.imread("assets/real/my_vtuber.png"); H,W = img.shape[:2]
d_raw = io.imread("assets/real/depth_anything_v2.png")
d = d_raw[...,0] if d_raw.ndim==3 else d_raw
d = transform.resize(d.astype(np.float32),(H,W),anti_aliasing=True,
                     preserve_range=True) if d.shape!=(H,W) else d.astype(np.float32)

border = np.concatenate([img[0,:],img[-1,:],img[:,0],img[:,-1]],0)
if border.ndim>1:
    bg=np.median(border,0); diff=np.linalg.norm(img.astype(np.float32)-bg,axis=-1)
else:
    bg=np.median(border); diff=np.abs(img.astype(np.float32)-bg)
similar=diff<40; lbl,_=ndimage.label(similar)
bl=np.concatenate([lbl[0,:],lbl[-1,:],lbl[:,0],lbl[:,-1]]); bl=bl[bl>0]
cm = ~(lbl == np.bincount(bl).argmax())

dd=np.where(cm,d,np.nan)
edges_q=np.percentile(dd[cm],np.linspace(0,100,N_BUCKETS+1))
edges_q[0]-=1e-6; edges_q[-1]+=1e-6
region_id=np.zeros((H,W),np.int32); nid=1
st=np.ones((3,3),np.uint8)
for b in range(N_BUCKETS):
    m=cm&(dd>=edges_q[b])&(dd<edges_q[b+1])
    L,k=ndimage.label(m,structure=st)
    for i in range(1,k+1):
        if int((L==i).sum())<MIN_REGION_SIZE: continue
        region_id[L==i]=nid; nid+=1
region_z=np.zeros(nid,np.float32)
for rid in range(1,nid):
    m=region_id==rid
    if m.any(): region_z[rid]=dd[m].mean()
v=region_z[1:]
if len(v) and v.max()>v.min(): region_z[1:]=(v-v.min())/(v.max()-v.min())

gray=color.rgb2gray(img[...,:3]) if img.ndim==3 else img.astype(np.float32)/255
edges_i=feature.canny(gray,sigma=CANNY_SIGMA,low_threshold=CANNY_LOW,high_threshold=CANNY_HIGH)
skel=morphology.skeletonize(edges_i)
pad=np.pad(skel.astype(np.uint8),1)
nbr=np.zeros((H,W),np.uint8)
for dy in range(3):
    for dx in range(3):
        if dy==1 and dx==1: continue
        nbr+=pad[dy:dy+H,dx:dx+W]
nbr*=skel.astype(np.uint8)
seg_lbl,n_seg = ndimage.label(skel & ~((nbr==3) & skel),
                              structure=np.ones((3,3),np.uint8))

_, idx_r = ndimage.distance_transform_edt(region_id==0, return_indices=True)
nearest_reg = region_id[idx_r[0], idx_r[1]]
skp = skel & (seg_lbl>0)
sids = seg_lbl[skp]; wts = region_z[nearest_reg[skp]]
seg_z = np.bincount(sids, weights=wts, minlength=n_seg+1) / np.maximum(np.bincount(sids, minlength=n_seg+1), 1)

npz = np.load("assets/real/tj_normalized/tj_normalized_accept.npz")
ys_c, xs_c = npz['ys'], npz['xs']

def find_cons(ys, xs):
    out=[]
    for i in range(len(ys)):
        y,x=int(ys[i]),int(xs[i])
        neigh=[]
        for dy,dx in DIRS:
            ny,nx=y+dy,x+dx
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
        si=[k for k in range(3) if k not in cb][0]
        segs=[]
        for k in [si,cb[0],cb[1]]:
            dy,dx=neigh[k]; ny,nx=y,x; hit=0
            for _ in range(WALK_LEN):
                ny+=dy; nx+=dx
                if not (0<=ny<H and 0<=nx<W): break
                ss=seg_lbl[ny,nx]
                if ss>0: hit=int(ss); break
            segs.append(hit)
        if all(s>0 for s in segs) and len(set(segs))==3:
            out.append(tuple(segs))
    return out
cons = find_cons(ys_c, xs_c)

# EDT for pixel → nearest segment
_, idx_s = ndimage.distance_transform_edt(seg_lbl==0, return_indices=True)
nearest_seg_yx = np.stack([idx_s[0].astype(np.int32), idx_s[1].astype(np.int32)], axis=-1)

# Dump
out = Path("assets/real/swift_cache"); out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out / "cache.npz",
    char_mask=cm,
    seg_lbl=seg_lbl.astype(np.int32),
    seg_z=seg_z.astype(np.float32),
    n_seg=np.int32(n_seg),
    cons=np.array(cons, np.int32),           # (N,3) → (stem, ca, cb)
    nearest_seg_yx=nearest_seg_yx,
    H=np.int32(H), W=np.int32(W),
)
print(f"[cache] {out/'cache.npz'}")
print(f"  H={H} W={W} n_seg={n_seg} cons={len(cons)}")
print(f"  seg_z nonzero: {int((seg_z[1:]>0).sum())}")
print(f"  char_mask px: {int(cm.sum())}")

# Expected output: canonical fusion result (for Swift comparison)
from collections import deque
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
EPS = np.float32(0.005)
eu = np.array([ca for s,ca,cb in cons] + [cb for s,ca,cb in cons], np.int32)
ev = np.array([s  for s,ca,cb in cons] + [s  for s,ca,cb in cons], np.int32)
adj = csr_matrix((np.ones(len(eu),np.uint8),(eu,ev)), shape=(n_seg+1,n_seg+1))
n_scc, scc_lbl = connected_components(adj, directed=True, connection='strong')

scc_u, scc_v = scc_lbl[eu], scc_lbl[ev]
in_deg = np.zeros(n_scc, np.int32); out_adj = [[] for _ in range(n_scc)]
for a,b in zip(scc_u, scc_v):
    a,b = int(a),int(b)
    if a != b: in_deg[b]+=1; out_adj[a].append(b)
q = deque([i for i in range(n_scc) if in_deg[i]==0]); topo=[]
while q:
    u=q.popleft(); topo.append(u)
    for vv in out_adj[u]:
        in_deg[vv]-=1
        if in_deg[vv]==0: q.append(vv)

scc_members = {}
for s in range(1, n_seg+1):
    scc_members.setdefault(int(scc_lbl[s]), []).append(s)

z = seg_z.copy()
for u_scc in topo:
    if u_scc not in scc_members: continue
    z_u = min(z[m] for m in scc_members[u_scc])
    for v_scc in out_adj[u_scc]:
        for mv in scc_members.get(v_scc, []):
            if z[mv] > z_u - EPS:
                z[mv] = z_u - EPS

zv = z[1:]
if zv.max() > zv.min():
    z_n = (z - zv.min()) / (zv.max() - zv.min())
else:
    z_n = z.copy()
z_n[0] = 0

# Pixel z
z_pix = z_n[seg_lbl]
z_pix = z_n[seg_lbl[nearest_seg_yx[...,0], nearest_seg_yx[...,1]]]
z_pix[~cm] = np.nan

np.savez_compressed(
    out / "expected.npz",
    z_n=z_n.astype(np.float32),         # per-segment
    z_pix=z_pix.astype(np.float32),     # per-pixel
    topo_order=np.array(topo, np.int32),
    scc_lbl=scc_lbl.astype(np.int32),
)
print(f"[expected] {out/'expected.npz'}  topo_len={len(topo)}")
