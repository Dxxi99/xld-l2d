#!/usr/bin/env python3
"""3way_nms.py — use NMS-filtered 343 T-junctions only."""
import numpy as np
from pathlib import Path
from collections import deque
from skimage import io, feature, morphology, color, transform
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

N_BUCKETS=7; MIN_REGION_SIZE=800; WALK_LEN=20
CANNY_SIGMA=1.5; CANNY_LOW=0.08; CANNY_HIGH=0.20
DIRS = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], np.int32)

img = io.imread("assets/real/my_vtuber.png"); H,W = img.shape[:2]
d_raw = io.imread("assets/real/depth_anything_v2.png")
d = d_raw[...,0] if d_raw.ndim==3 else d_raw
d = transform.resize(d.astype(np.float32), (H,W), anti_aliasing=True,
                     preserve_range=True) if d.shape!=(H,W) else d.astype(np.float32)

border = np.concatenate([img[0,:],img[-1,:],img[:,0],img[:,-1]],0)
if border.ndim>1:
    bg=np.median(border,0); diff=np.linalg.norm(img.astype(np.float32)-bg,axis=-1)
else:
    bg=np.median(border); diff=np.abs(img.astype(np.float32)-bg)
similar=diff<40; lbl,_=ndimage.label(similar)
bl=np.concatenate([lbl[0,:],lbl[-1,:],lbl[:,0],lbl[:,-1]]); bl=bl[bl>0]
cm = ~(lbl == np.bincount(bl).argmax())

dd = np.where(cm,d,np.nan)
edges = np.percentile(dd[cm], np.linspace(0,100,N_BUCKETS+1))
edges[0]-=1e-6; edges[-1]+=1e-6
region_id = np.zeros((H,W),np.int32); nid=1
st=np.ones((3,3),np.uint8)
for b in range(N_BUCKETS):
    m = cm & (dd>=edges[b]) & (dd<edges[b+1])
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
print(f"[regions] {nid-1}")

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
seg_lbl, n_seg = ndimage.label(skel & ~((nbr==3) & skel),
                               structure=np.ones((3,3),np.uint8))
print(f"[segments] {n_seg}")

_, idx = ndimage.distance_transform_edt(region_id==0, return_indices=True)
nearest_reg = region_id[idx[0], idx[1]]
skp = skel & (seg_lbl>0)
sids = seg_lbl[skp]; wts = region_z[nearest_reg[skp]]
seg_z = np.bincount(sids, weights=wts, minlength=n_seg+1) / np.maximum(np.bincount(sids, minlength=n_seg+1), 1)

seg_region = np.zeros(n_seg+1, np.int32)
for s in range(1,n_seg+1):
    m=(seg_lbl==s)&skp
    if m.any(): seg_region[s]=np.bincount(nearest_reg[m]).argmax()

# Load NMS T-junctions
npz = np.load("assets/real/tj_normalized/tj_normalized_accept.npz")
ys_clean, xs_clean = npz['ys'], npz['xs']
print(f"[tj clean] {len(ys_clean)}")

def find_cons(ys, xs):
    cons=[]
    for i in range(len(ys)):
        y,x = int(ys[i]), int(xs[i])
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
        if all(s>0 for s in segs): cons.append(tuple(segs))
    return cons

cons = find_cons(ys_clean, xs_clean)
distinct = [c for c in cons if len(set(c))==3]
print(f"[constraints] total={len(cons)} distinct={len(distinct)}")

def viol(sz, cons):
    return sum(1 for s,ca,cb in cons if sz[s] > min(sz[ca],sz[cb]) + 0.01)
n = len(distinct)
print(f"[initial] viol={viol(seg_z, distinct)}/{n} ({100*viol(seg_z,distinct)/n:.1f}%)")

# SCC topo
u_list=[]; v_list=[]
for s,ca,cb in distinct:
    u_list += [ca, cb]; v_list += [s, s]
eu=np.array(u_list,np.int32); ev=np.array(v_list,np.int32)
adj=csr_matrix((np.ones(len(eu),np.uint8),(eu,ev)),shape=(n_seg+1,n_seg+1))
n_scc, scc_lbl = connected_components(adj, directed=True, connection='strong')
print(f"[scc] {n_scc}")

scc_u, scc_v = scc_lbl[eu], scc_lbl[ev]
cond = set((int(a),int(b)) for a,b in zip(scc_u,scc_v) if a!=b)
in_deg=np.zeros(n_scc,np.int32); out_adj=[[] for _ in range(n_scc)]
for a,b in cond: in_deg[b]+=1; out_adj[a].append(b)
q=deque([i for i in range(n_scc) if in_deg[i]==0]); topo_order=[]
while q:
    u=q.popleft(); topo_order.append(u)
    for vv in out_adj[u]:
        in_deg[vv]-=1
        if in_deg[vv]==0: q.append(vv)
layer_of_scc=np.zeros(n_scc,np.int32)
for i,scc in enumerate(topo_order): layer_of_scc[scc]=i
max_l=max(int(layer_of_scc.max()),1)
topo_z=np.zeros(n_seg+1,np.float32)
for s in range(1,n_seg+1):
    topo_z[s] = 1.0 - layer_of_scc[scc_lbl[s]]/max_l

def colorize(z):
    z=np.clip(z,0,1)
    r=np.clip(1.5-np.abs(z-0.75)*4,0,1)
    g=np.clip(1.5-np.abs(z-0.50)*4,0,1)
    b=np.clip(1.5-np.abs(z-0.25)*4,0,1)
    return (np.stack([r,g,b],-1)*255).astype(np.uint8)

out=Path("assets/real/3way_nms"); out.mkdir(parents=True, exist_ok=True)
_, idx2 = ndimage.distance_transform_edt(seg_lbl==0, return_indices=True)
for name, sz in [('A',seg_z),('B',topo_z),('C',0.5*seg_z+0.5*topo_z)]:
    zmap = sz[seg_lbl[idx2[0],idx2[1]]]
    vv = zmap[cm]
    if len(vv) and vv.max()>vv.min():
        zmap = (zmap-vv.min())/(vv.max()-vv.min())
    vis = img[...,:3].copy()
    if vis.dtype != np.uint8: vis = np.clip(vis*255,0,255).astype(np.uint8)
    vis[cm] = colorize(zmap)[cm]
    io.imsave(out/f"{name}_zmap.png", vis)
    print(f"[{name}] viol={viol(sz,distinct)}/{n} z=[{sz[1:].min():.3f},{sz[1:].max():.3f}]")
print(f"[save] {out}/")
