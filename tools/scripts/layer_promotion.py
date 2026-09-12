#!/usr/bin/env python3
"""layer_promotion.py — canonical fusion (topo+depth) → N layers → occlusion graph."""
import numpy as np
from pathlib import Path
from collections import deque
from skimage import io, feature, morphology, color, transform
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

N_BUCKETS=7; MIN_REGION_SIZE=800; WALK_LEN=20; N_LAYERS=7; EPS=0.005
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
edges=np.percentile(dd[cm],np.linspace(0,100,N_BUCKETS+1))
edges[0]-=1e-6; edges[-1]+=1e-6
region_id=np.zeros((H,W),np.int32); nid=1
st=np.ones((3,3),np.uint8)
for b in range(N_BUCKETS):
    m=cm&(dd>=edges[b])&(dd<edges[b+1])
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
seg_lbl,n_seg = ndimage.label(skel & ~((nbr==3) & skel),
                              structure=np.ones((3,3),np.uint8))
print(f"[segments] {n_seg}")

_, idx = ndimage.distance_transform_edt(region_id==0, return_indices=True)
nearest_reg = region_id[idx[0], idx[1]]
skp = skel & (seg_lbl>0)
sids = seg_lbl[skp]; wts = region_z[nearest_reg[skp]]
seg_z = np.bincount(sids, weights=wts, minlength=n_seg+1) / np.maximum(np.bincount(sids, minlength=n_seg+1), 1)

# --- Load clean T-junctions ---
npz = np.load("assets/real/tj_normalized/tj_normalized_accept.npz")
ys_c, xs_c = npz['ys'], npz['xs']
print(f"[tj clean] {len(ys_c)}")

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
print(f"[constraints] {len(cons)}")

# --- Canonical fusion: topo order, single pass ---
eu = np.array([ca for s,ca,cb in cons] + [cb for s,ca,cb in cons], np.int32)
ev = np.array([s for s,ca,cb in cons] + [s for s,ca,cb in cons], np.int32)
adj = csr_matrix((np.ones(len(eu),np.uint8),(eu,ev)), shape=(n_seg+1,n_seg+1))
n_scc, scc_lbl = connected_components(adj, directed=True, connection='strong')
print(f"[scc] {n_scc} (n_seg={n_seg})")

scc_u, scc_v = scc_lbl[eu], scc_lbl[ev]
in_deg = np.zeros(n_scc, np.int32); out_adj = [[] for _ in range(n_scc)]
for a,b in zip(scc_u, scc_v):
    a, b = int(a), int(b)
    if a != b: in_deg[b]+=1; out_adj[a].append(b)
q = deque([i for i in range(n_scc) if in_deg[i]==0]); topo=[]
while q:
    u=q.popleft(); topo.append(u)
    for vv in out_adj[u]:
        in_deg[vv]-=1
        if in_deg[vv]==0: q.append(vv)

# Map each constraint's stem & crossbars to SCC id
cons_scc = [(scc_lbl[s], scc_lbl[ca], scc_lbl[cb]) for s,ca,cb in cons]

# Process segments in topo order (source first), pull stems down
z = seg_z.copy()
for scc in topo:
    # find all constraints where this scc is the stem
    for cs, cca, ccb in cons_scc:
        if cs != scc: continue
        target = min(z[cca] if cca!=cs else z[cca],
                     z[ccb] if ccb!=cs else z[ccb]) - EPS
        # apply to every segment in this scc (usually just one)
        members = np.where(scc_lbl == scc)[0]
        for m in members:
            if m == 0: continue
            if z[m] > target: z[m] = target

# Normalize
zv = z[1:]
if zv.max() > zv.min():
    z_n = (z - zv.min())/(zv.max()-zv.min())
else:
    z_n = z
z_n[0] = 0

# Verify
n_viol = sum(1 for s,ca,cb in cons if z_n[s] > min(z_n[ca],z_n[cb]) + EPS)
print(f"[canonical fusion] viol={n_viol}/{len(cons)}")

# --- Threshold into N layers ---
z_pix = z_n[seg_lbl]
z_pix[~cm] = 0
# assign by quantile of z among char pixels
zvals = z_pix[cm]
q_edges = np.quantile(zvals, np.linspace(0,1,N_LAYERS+1))
q_edges[-1]+=1e-6
layer_id = np.zeros((H,W),np.uint8)
for l in range(N_LAYERS):
    m = cm & (z_pix >= q_edges[l]) & (z_pix < q_edges[l+1])
    layer_id[m] = l

print(f"[layers] N={N_LAYERS}")
for l in range(N_LAYERS):
    m = layer_id == l
    print(f"  L{l}: {int(m.sum()):>8} px  z=[{q_edges[l]:.3f},{q_edges[l+1]:.3f})")

# --- Save layer masks ---
out = Path("assets/real/layers"); out.mkdir(parents=True, exist_ok=True)
palette = np.array([
    [  40,  40, 180],  # L0 dark blue   (far)
    [  60, 100, 200],  # L1
    [ 100, 160, 200],  # L2
    [ 180, 180, 180],  # L3 gray        (mid)
    [ 220, 160, 140],  # L4
    [ 230, 100,  90],  # L5
    [ 240,  50,  50],  # L6 red         (near)
], dtype=np.uint8)

vis_layers = img[...,:3].copy()
if vis_layers.dtype != np.uint8:
    vis_layers = np.clip(vis_layers*255,0,255).astype(np.uint8)
for l in range(N_LAYERS):
    m = layer_id == l
    if not m.any(): continue
    # Save individual mask (binary)
    mask_vis = np.zeros((H,W,3), np.uint8)
    mask_vis[m] = palette[l]
    io.imsave(out / f"L{l}.png", mask_vis)
    # Overlay
    vis_layers[m] = (vis_layers[m].astype(np.float32)*0.3 +
                     palette[l].astype(np.float32)*0.7).astype(np.uint8)
io.imsave(out / "layers_overlay.png", vis_layers)

# --- Occlusion graph (adjacency) ---
# For each pair of adjacent layers (l, l+1), report boundary pixel count
occl = np.zeros((N_LAYERS, N_LAYERS), np.int32)
# Adjacency from constraints: crossbar occludes stem
for s, ca, cb in cons:
    ls = layer_id[skp & (seg_lbl==s)].max() if (skp & (seg_lbl==s)).any() else 0
    la = layer_id[skp & (seg_lbl==ca)].max() if (skp & (seg_lbl==ca)).any() else 0
    lb = layer_id[skp & (seg_lbl==cb)].max() if (skp & (seg_lbl==cb)).any() else 0
    if la > ls: occl[la, ls] += 1
    if lb > ls: occl[lb, ls] += 1

print()
print(f"[occlusion graph] (row occludes col)")
print("     " + "".join(f"L{j:<5d}" for j in range(N_LAYERS)))
for i in range(N_LAYERS):
    row = "".join(f"{occl[i,j]:<6d}" for j in range(N_LAYERS))
    print(f"L{i}: {row}")

print()
print(f"[save] {out}/L*.png  layers_overlay.png")
