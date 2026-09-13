#!/usr/bin/env python3
"""layer_promotion_v2.py — canonical fusion + EDT layer spread + Swift cache dump

Usage:
    python3 layer_promotion_v2.py [IMG_png] [DEPTH_png] [TJ_npz] [OUT_LAYERS_dir] [SWIFT_CACHE_dir]
"""
import sys
import numpy as np
from pathlib import Path
from collections import deque
from skimage import io, feature, morphology, color, transform
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

N_BUCKETS=7; MIN_REGION_SIZE=800; WALK_LEN=20; N_LAYERS=7; EPS=np.float32(0.005)
CANNY_SIGMA=1.5; CANNY_LOW=0.08; CANNY_HIGH=0.20
DIRS = np.array([(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)], np.int32)

# ---- CLI ----
IMG_ARG   = sys.argv[1] if len(sys.argv) > 1 else None
DEPTH_ARG = sys.argv[2] if len(sys.argv) > 2 else None
TJ_ARG    = sys.argv[3] if len(sys.argv) > 3 else None
LAYERS_ARG= sys.argv[4] if len(sys.argv) > 4 else None
CACHE_ARG = sys.argv[5] if len(sys.argv) > 5 else None
CHAR_MASK_ARG = sys.argv[6] if len(sys.argv) > 6 else None

IMG_PATH   = Path(IMG_ARG).expanduser()   if IMG_ARG   else (Path.home()/"XLD-L2D/assets/real/my_vtuber.png")
DEPTH_PATH = Path(DEPTH_ARG).expanduser() if DEPTH_ARG else (Path.home()/"XLD-L2D/assets/real/depth_anything_v2.png")
TJ_PATH    = Path(TJ_ARG).expanduser()    if TJ_ARG    else (Path.home()/"XLD-L2D/assets/real/tj_normalized/tj_normalized_accept.npz")
OUT_LAYERS = Path(LAYERS_ARG).expanduser()if LAYERS_ARG else (Path.home()/"XLD-L2D/assets/real/layers_v2")
SWIFT_DIR  = Path(CACHE_ARG).expanduser() if CACHE_ARG else (Path.home()/"XLD-L2D/assets/real/swift_cache")

for p in [IMG_PATH, DEPTH_PATH, TJ_PATH]:
    if not p.exists():
        print(f"[error] not found: {p}"); sys.exit(1)

OUT_LAYERS.mkdir(parents=True, exist_ok=True)
SWIFT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[cfg] IMG={IMG_PATH}")
print(f"[cfg] DEPTH={DEPTH_PATH}")
print(f"[cfg] TJ={TJ_PATH}")
print(f"[cfg] OUT_LAYERS={OUT_LAYERS}")
print(f"[cfg] SWIFT_CACHE={SWIFT_DIR}")

# ---- load ----
img = io.imread(str(IMG_PATH)); H,W = img.shape[:2]
d_raw = io.imread(str(DEPTH_PATH))
d = d_raw[...,0] if d_raw.ndim==3 else d_raw
d = transform.resize(d.astype(np.float32),(H,W),anti_aliasing=True,
                     preserve_range=True) if d.shape!=(H,W) else d.astype(np.float32)
print(f"[img] H={H} W={W}")

# ---- char mask ----
# 우선순위: --char-mask 인자 (ToonOut) > fallback (border flood fill)
cm = None
if CHAR_MASK_ARG is not None:
    cm_path = Path(CHAR_MASK_ARG)
    if not cm_path.is_absolute():
        cm_path = Path.cwd() / cm_path
    if cm_path.exists():
        from PIL import Image as _PILImage
        cm_img = _PILImage.open(str(cm_path)).convert("L")
        cm_np = np.array(cm_img, dtype=np.float32) / 255.0
        if cm_np.shape != (H, W):
            cm_np = transform.resize(cm_np, (H, W), anti_aliasing=True, preserve_range=True)
        # 임계값 0.5 (이진), 안티앨리어싱 유지
        cm = (cm_np > 0.5)
        print(f"[char_mask] loaded from {cm_path.name}  ratio={cm.mean():.3f}")
    else:
        print(f"[char_mask] NOT FOUND: {cm_path}  → fallback")

if cm is None:
    # fallback: border-connected white flood fill
    print("[char_mask] using fallback (border flood fill)")
    rgb_i = img[...,:3].astype(np.int32) if img.ndim == 3 else img.astype(np.int32)[...,None].repeat(3, axis=-1)
    white = (rgb_i[...,0] >= 245) & (rgb_i[...,1] >= 245) & (rgb_i[...,2] >= 245)
    white = ndimage.binary_closing(white, iterations=2)
    lbl, _ = ndimage.label(white)
    bl = np.concatenate([lbl[0,:], lbl[-1,:], lbl[:,0], lbl[:,-1]])
    bl = bl[bl > 0]
    if len(bl) > 0:
        bg_label = np.bincount(bl).argmax()
        bg = (lbl == bg_label)
    else:
        bg = white
    cm = ~bg
    cm = ndimage.binary_closing(cm, iterations=3)
    cm = ndimage.binary_opening(cm, iterations=2)
    cm = ndimage.binary_fill_holes(cm)
    cm = ndimage.binary_closing(cm, iterations=3)
    cm = ndimage.binary_opening(cm, iterations=1)
    cm = ndimage.binary_fill_holes(cm)

# ---- region-based depth buckets ----
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
print(f"[regions] {nid-1}")

# ---- T-junction based segments ----
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

_, idx_r = ndimage.distance_transform_edt(region_id==0, return_indices=True)
nearest_reg = region_id[idx_r[0], idx_r[1]]
skp = skel & (seg_lbl>0)
sids = seg_lbl[skp]; wts = region_z[nearest_reg[skp]]
seg_z = np.bincount(sids, weights=wts, minlength=n_seg+1) / np.maximum(np.bincount(sids, minlength=n_seg+1), 1)

# ---- load T-junction candidates ----
npz = np.load(str(TJ_PATH))
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
print(f"[constraints] {len(cons)}")

# ---- canonical fusion (SCC + topo + single pass) ----
eu = np.array([ca for s,ca,cb in cons] + [cb for s,ca,cb in cons], np.int32)
ev = np.array([s  for s,ca,cb in cons] + [s  for s,ca,cb in cons], np.int32)
adj = csr_matrix((np.ones(len(eu),np.uint8),(eu,ev)), shape=(n_seg+1,n_seg+1))
n_scc, scc_lbl = connected_components(adj, directed=True, connection='strong')
print(f"[scc] {n_scc}")

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
print(f"[topo] {len(topo)}/{n_scc}")

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

nv = sum(1 for s,ca,cb in cons if z[s] > min(z[ca], z[cb]) + 1e-6)
print(f"[canonical fusion] viol={nv}/{len(cons)}")

zv = z[1:]
if zv.max() > zv.min():
    z_n = (z - zv.min()) / (zv.max() - zv.min())
else:
    z_n = z.copy()
z_n[0] = 0

_, idx_s = ndimage.distance_transform_edt(seg_lbl==0, return_indices=True)
nearest_seg = seg_lbl[idx_s[0], idx_s[1]]
z_pix = z_n[nearest_seg]
z_pix[~cm] = np.nan

zvals = z_pix[cm]
print(f"[z_pix] char z range {zvals.min():.3f}..{zvals.max():.3f}")
q_edges = np.quantile(zvals, np.linspace(0,1,N_LAYERS+1))
q_edges[0] -= 1e-6; q_edges[-1] += 1e-6

layer_id = np.full((H,W), 255, np.uint8)
for l in range(N_LAYERS):
    m = cm & (z_pix >= q_edges[l]) & (z_pix < q_edges[l+1])
    layer_id[m] = l

print(f"[layers] N={N_LAYERS}")
for l in range(N_LAYERS):
    m = layer_id == l
    print(f"  L{l}: {int(m.sum()):>8} px")

# ---- save layer masks + overlay ----
palette = np.array([
    [  40,  40, 200],[  60, 110, 220],[ 100, 160, 220],[ 180, 180, 180],
    [ 220, 160, 140],[ 230, 100,  90],[ 240,  50,  50],
], dtype=np.uint8)
vis_l = img[...,:3].copy()
if vis_l.dtype != np.uint8: vis_l = np.clip(vis_l*255,0,255).astype(np.uint8)
for l in range(N_LAYERS):
    m = layer_id == l
    if not m.any(): continue
    mask_vis = np.zeros((H,W,3), np.uint8); mask_vis[m] = palette[l]
    io.imsave(OUT_LAYERS / f"L{l}.png", mask_vis)
    vis_l[m] = (vis_l[m].astype(np.float32)*0.3 + palette[l].astype(np.float32)*0.7).astype(np.uint8)
io.imsave(OUT_LAYERS / "layers_overlay.png", vis_l)
print(f"[save] {OUT_LAYERS}/")

# ---- save Swift cache ----
cm.astype(np.uint8).tofile(SWIFT_DIR / "char_mask.bin")
seg_lbl.astype(np.int32).tofile(SWIFT_DIR / "seg_lbl.bin")
z_n.astype(np.float32).tofile(SWIFT_DIR / "seg_z.bin")
cons_arr = np.array(cons, dtype=np.int32) if cons else np.zeros((0,3),np.int32)
cons_arr.tofile(SWIFT_DIR / "cons.bin")
layer_id.tofile(SWIFT_DIR / "expected_layer_id.bin")
nearest_seg.astype(np.int32).tofile(SWIFT_DIR / "nearest_seg.bin")

with open(SWIFT_DIR / "meta.txt", "w") as f:
    f.write(f"H {H}\n")
    f.write(f"W {W}\n")
    f.write(f"n_seg {n_seg}\n")
    f.write(f"n_cons {len(cons)}\n")

print(f"[swift_cache] {SWIFT_DIR}")
print(f"  char_mask.bin  {cm.size}")
print(f"  seg_lbl.bin    {seg_lbl.size}")
print(f"  seg_z.bin      {z_n.size}")
print(f"  cons.bin       {cons_arr.size}")
print(f"  layer_id.bin   {layer_id.size}")
print(f"  nearest_seg.bin {nearest_seg.size}")
print(f"  meta.txt       H={H} W={W} n_seg={n_seg} n_cons={len(cons)}")
