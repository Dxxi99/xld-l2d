import numpy as np
from PIL import Image
from scipy import ndimage
import os

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
depth = Image.open(f"{ROOT}/depth_anything_v2.png").convert("L")
rgb = np.array(img).astype(np.float32)
d   = np.array(depth).astype(np.float32)
H, W = rgb.shape[:2]
print("image %dx%d" % (W, H))

# ---- char mask (border-connected light = bg) ----
R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
light = (R > 228) & (G > 228) & (B > 228)
lab, n = ndimage.label(light)
border = set(lab[0,:].tolist()) | set(lab[-1,:].tolist()) \
       | set(lab[:,0].tolist()) | set(lab[:,-1].tolist())
border.discard(0)
bg = np.isin(lab, list(border))
char = ~bg
char = ndimage.binary_closing(char, iterations=2)
char = ndimage.binary_opening(char, iterations=1)
char = ndimage.binary_fill_holes(char)
print("char: %.2f%%" % (100.0*char.sum()/char.size))

# ---- A) 픽셀 단위 depth on char (구간화 없음) ----
vis_a = (rgb * 0.35 + 50).astype(np.uint8)
cd = d[char]
dmin, dmax = cd.min(), cd.max()
dn = np.zeros((H, W), np.float32)
dn[char] = (d[char] - dmin) / (dmax - dmin + 1e-6)
r_ch = (dn * 255).astype(np.uint8)
b_ch = ((1 - dn) * 255).astype(np.uint8)
tint = np.stack([r_ch, np.full_like(r_ch, 40), b_ch], -1)
vis_a[char] = (0.85 * tint[char] + 0.15 * vis_a[char]).astype(np.uint8)
Image.fromarray(vis_a).save(f"{ROOT}/rgn_pixelwise.png")
print("saved rgn_pixelwise.png")

# ---- B) depth quantile slicing within char ----
qs = np.percentile(cd, [15, 30, 45, 60, 75, 90])
print("quantiles:", [round(q,1) for q in qs])

bucket = np.full((H, W), -1, np.int8)
bucket[char] = np.digitize(d[char], qs)

MIN_SIZE = 800
regions = np.zeros((H, W), np.int32)
next_id = 0
region_info = []
for b in range(7):
    m = char & (bucket == b)
    lab_b, n_b = ndimage.label(m, structure=np.ones((3,3)))
    for i in range(1, n_b+1):
        mm = lab_b == i
        sz = int(mm.sum())
        if sz < MIN_SIZE: continue
        next_id += 1
        regions[mm] = next_id
        region_info.append((next_id, float(d[mm].mean()), sz, b))

print("regions (>=%d px): %d" % (MIN_SIZE, next_id))

# unassigned char → nearest region via EDT
unassigned = char & (regions == 0)
print("unassigned: %.2f%% of char" % (100.0*unassigned.sum()/char.sum()))
ind = ndimage.distance_transform_edt(regions == 0,
        return_distances=False, return_indices=True)
regions = regions[tuple(ind)]
regions[~char] = 0

# region means
region_z = np.zeros(next_id + 1, np.float32)
region_sz = np.zeros(next_id + 1, np.int32)
for (rid, mz, sz, b) in region_info:
    region_z[rid] = mz
    region_sz[rid] = sz

if next_id > 0:
    valid = region_sz[1:] > 0
    rmin = region_z[1:][valid].min()
    rmax = region_z[1:][valid].max()
    rnorm = (region_z - rmin) / (rmax - rmin + 1e-6)
    print("region depth: %.1f .. %.1f" % (rmin, rmax))

    # render
    vis_b = (rgb * 0.35 + 50).astype(np.uint8)
    for rid in range(1, next_id+1):
        m = regions == rid
        if not m.any(): continue
        z = float(rnorm[rid])
        r = int(z * 255)
        bl = int((1-z) * 255)
        vis_b[m] = (0.9 * np.array([r, 40, bl], np.float32) + 0.1 * vis_b[m]).astype(np.uint8)
    Image.fromarray(vis_b).save(f"{ROOT}/rgn_sliced.png")
    print("saved rgn_sliced.png")

    for name, box in [("face",(400,500,1000,1100)),
                      ("hair",(200,300,1500,1000)),
                      ("body",(600,1200,1400,1800))]:
        x0,y0,x1,y1 = box
        Image.fromarray(vis_b[y0:y1, x0:x1]).save(f"{ROOT}/rgn_sliced_{name}.png")
    print("saved crops")

# ---- C) 특정 영역의 depth 통계 (의미 검증) ----
print("\n[region depth 샘플]")
for name, (x0,y0,x1,y1) in [
    ("face_skin",  (500, 600, 850, 1000)),
    ("front_hair", (600, 350, 900, 550)),
    ("body_top",   (600, 1200, 1000, 1400)),
    ("left_knee",  (300, 1400, 550, 1700)),
    ("right_shoe", (900, 2100, 1200, 2400)),
]:
    box_depth = d[y0:y1, x0:x1]
    box_char = char[y0:y1, x0:x1]
    vals = box_depth[box_char]
    if len(vals):
        print("  %-12s n=%6d  mean=%.1f  std=%.1f" % (
            name, len(vals), vals.mean(), vals.std()))
    else:
        print("  %-12s (no char)" % name)
