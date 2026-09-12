import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.morphology import skeletonize
from scipy import ndimage
import os
from collections import deque

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
dep = Image.open(f"{ROOT}/depth_anything_v2.png").convert("L")
rgb = np.array(img).astype(np.float32)
d   = np.array(dep).astype(np.float32)
H, W = rgb.shape[:2]
print("image %dx%d" % (W, H))

# char mask
R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
light = (R>228) & (G>228) & (B>228)
lab_bg, _ = ndimage.label(light)
border = set(lab_bg[0,:].tolist())|set(lab_bg[-1,:].tolist())\
        |set(lab_bg[:,0].tolist())|set(lab_bg[:,-1].tolist())
border.discard(0)
char = ~np.isin(lab_bg, list(border))
char = ndimage.binary_closing(char, iterations=2)
char = ndimage.binary_opening(char, iterations=1)
char = ndimage.binary_fill_holes(char)

# regions
cd = d[char]
qs = np.percentile(cd, [15,30,45,60,75,90])
bucket = np.full((H,W), -1, np.int8)
bucket[char] = np.digitize(d[char], qs)
regions = np.zeros((H,W), np.int32)
next_id = 0
rmean = []
for b in range(7):
    m = char & (bucket == b)
    lb, nb = ndimage.label(m, structure=np.ones((3,3)))
    for i in range(1, nb+1):
        mm = lb == i
        if mm.sum() < 800: continue
        next_id += 1
        regions[mm] = next_id
        rmean.append(float(d[mm].mean()))
# unassigned fill
ind = ndimage.distance_transform_edt(regions==0, return_distances=False, return_indices=True)
regions = regions[tuple(ind)]
regions[~char] = 0
print("regions: %d" % next_id)

region_z = np.zeros(next_id+1, np.float32)
region_z[1:] = np.array(rmean, np.float32)
rmin, rmax = region_z[1:].min(), region_z[1:].max()
region_zn = np.zeros_like(region_z)
region_zn[1:] = (region_z[1:]-rmin)/(rmax-rmin+1e-6)
print("region depth: %.1f .. %.1f" % (rmin, rmax))

# skeleton + T
gray = 0.299*rgb[...,0]+0.587*rgb[...,1]+0.114*rgb[...,2]
edge = canny(gray/255.0, sigma=1.5, low_threshold=0.08, high_threshold=0.20)
skel = skeletonize(edge)
print("skeleton: %d px" % skel.sum())

DIRS = [(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)]
DV = np.array(DIRS, dtype=np.float32)
skel_u8 = skel.astype(np.uint8)
def sh(a, dy, dx):
    out = np.zeros_like(a)
    yd = slice(max(0,dy), H+min(0,dy)); xd = slice(max(0,dx), W+min(0,dx))
    ys_ = slice(max(0,-dy), H+min(0,-dy)); xs_ = slice(max(0,-dx), W+min(0,-dx))
    out[yd,xd] = a[ys_,xs_]
    return out
n_neigh = np.zeros((H,W), np.uint8)
for dy,dx in DIRS: n_neigh += sh(skel_u8, dy, dx)
n_neigh[~skel] = 0
branch = skel & (n_neigh == 3)
ys, xs = np.where(branch)
T_list = []
for y,x in zip(ys,xs):
    ndirs = []
    for i,(dy,dx) in enumerate(DIRS):
        yy,xx = y+dy, x+dx
        if 0<=yy<H and 0<=xx<W and skel[yy,xx]: ndirs.append(i)
    if len(ndirs) != 3: continue
    bp=None; bd=2.0
    for a in range(3):
        for b in range(a+1,3):
            dot = float(np.dot(DV[ndirs[a]], DV[ndirs[b]]))
            if dot < bd: bd=dot; bp=(a,b)
    si = [i for i in range(3) if i not in bp][0]
    T_list.append((int(y),int(x),int(ndirs[si]),
                   int(ndirs[bp[0]]), int(ndirs[bp[1]])))

used = np.zeros((H,W), bool)
T_final = []
for t in T_list:
    y,x = t[0],t[1]
    if used[y,x]: continue
    y0,y1 = max(0,y-3),min(H,y+4); x0,x1 = max(0,x-3),min(W,x+4)
    if used[y0:y1,x0:x1].any(): continue
    T_final.append(t); used[y0:y1,x0:x1] = True
print("T after NMS: %d" % len(T_final))

t_mask = np.zeros((H,W), bool)
for (y,x,*_) in T_final:
    y0,y1 = max(0,y-2),min(H,y+3); x0,x1 = max(0,x-2),min(W,x+3)
    t_mask[y0:y1,x0:x1] = True
seg_skel = skel & ~t_mask
seg_lab, n_seg = ndimage.label(seg_skel, structure=np.ones((3,3), bool))
print("segments: %d" % n_seg)

# --- FAST seg z_prior via EDT ---
# For every pixel, nearest segment id and distance
seg_ind = ndimage.distance_transform_edt(seg_lab==0, return_distances=False, return_indices=True)
nearest_seg = seg_lab[tuple(seg_ind)]
# only within char
within = char & (nearest_seg > 0)
seg_sum = np.zeros(n_seg+1, np.float64)
seg_cnt = np.zeros(n_seg+1, np.int64)
np.add.at(seg_sum, nearest_seg[within], region_zn[regions[within]])
np.add.at(seg_cnt, nearest_seg[within], 1)
seg_z = np.zeros(n_seg+1, np.float32)
valid_seg = seg_cnt > 0
seg_z[valid_seg] = (seg_sum[valid_seg] / seg_cnt[valid_seg]).astype(np.float32)
print("segments with prior: %d / %d" % (int(valid_seg.sum()), n_seg))

# constraints
def find_seg(y, x, dy, dx, mw=20):
    for k in range(1, mw+1):
        yy,xx = y+k*dy, x+k*dx
        if not (0<=yy<H and 0<=xx<W): break
        if seg_lab[yy,xx] > 0: return int(seg_lab[yy,xx])
    v0 = np.array([dy,dx],np.float32); v0 /= (np.linalg.norm(v0)+1e-6)
    q = deque([(y+dy,x+dx,1)]); seen={(y+dy,x+dx)}
    while q:
        cy,cx,dd = q.popleft()
        if dd>mw: continue
        if seg_lab[cy,cx] > 0: return int(seg_lab[cy,cx])
        for ddy,ddx in DIRS:
            ny,nx = cy+ddy, cx+ddx
            if (ny,nx) in seen: continue
            if not (0<=ny<H and 0<=nx<W): continue
            vy,vx = ny-y, nx-x
            nn = np.sqrt(vy*vy+vx*vx)+1e-6
            if (vy*v0[0]+vx*v0[1])/nn < 0.5: continue
            seen.add((ny,nx))
            if skel[ny,nx] or seg_lab[ny,nx]>0:
                q.append((ny,nx,dd+1))
    return 0

cstem,ccross = [],[]
for (y,x,si,ca,cb) in T_final:
    ss = find_seg(y,x,*DIRS[si])
    sa = find_seg(y,x,*DIRS[ca])
    sb = find_seg(y,x,*DIRS[cb])
    if ss>0 and sa>0 and sb>0 and ss!=sa and ss!=sb:
        cstem += [ss,ss]; ccross += [sa,sb]
cstem = np.array(cstem, np.int32); ccross = np.array(ccross, np.int32)
print("constraints: %d" % len(cstem))

# relaxation
z = seg_z.copy(); prior = seg_z.copy()
MARGIN=0.03; LAMBDA=0.5; MU=0.02
for it in range(300):
    a = z[cstem]; b = z[ccross]
    viol = a > (b - MARGIN)
    if not viol.any(): break
    delta = (a - (b - MARGIN)) * viol
    np.add.at(z, cstem, -LAMBDA*0.5*delta)
    np.add.at(z, ccross, +LAMBDA*0.5*delta)
    z[valid_seg] += MU * (prior[valid_seg] - z[valid_seg])
print("iterations: %d" % (it+1))
print("violations: %d / %d" % (int((z[cstem]>z[ccross]).sum()), len(cstem)))
diff = z - prior
print("adjust mean %.3f max %.3f >0.05: %d" % (
    np.abs(diff[valid_seg]).mean(), np.abs(diff[valid_seg]).max(),
    int((np.abs(diff[valid_seg])>0.05).sum())))

# render
zmn, zmx = z[valid_seg].min(), z[valid_seg].max()
zn = np.zeros_like(z); zn[valid_seg] = (z[valid_seg]-zmn)/(zmx-zmn+1e-6)

vis = (rgb*0.35+50).astype(np.uint8)
for s in range(1,n_seg+1):
    if not valid_seg[s]: continue
    m = seg_lab == s
    if not m.any(): continue
    r = int(zn[s]*255); bl = int((1-zn[s])*255)
    vis[m] = (0.9*np.array([r,40,bl],np.float32) + 0.1*vis[m]).astype(np.uint8)
for (y,x,*_) in T_final:
    y0,y1 = max(0,y-1),min(H,y+2); x0,x1 = max(0,x-1),min(W,x+2)
    vis[y0:y1,x0:x1] = [255,255,255]

Image.fromarray(vis).save(f"{ROOT}/fusion_v3.png")
for nm, box in [("face",(400,500,1000,1100)),("hair",(200,300,1500,1000)),("body",(600,1200,1400,1800))]:
    x0,y0,x1,y1 = box
    Image.fromarray(vis[y0:y1,x0:x1]).save(f"{ROOT}/fusion_v3_{nm}.png")
print("saved fusion_v3*.png")
