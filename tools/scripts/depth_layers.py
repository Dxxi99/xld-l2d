import numpy as np
from PIL import Image
import os
try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARN: scipy missing -> pip install scipy")

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img   = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
depth = Image.open(f"{ROOT}/depth_anything_v2.png").convert("L")

rgb = np.array(img).astype(np.float32)
d   = np.array(depth).astype(np.float32)
H, W = rgb.shape[:2]
print("image", img.size)

# ============================================================
# CHAR MASK — border-connected light pixels = bg
# ============================================================
R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
light = (R > 228) & (G > 228) & (B > 228)

if HAS_SCIPY:
    lab, n = ndimage.label(light)
    border = set(lab[0,:].tolist()) | set(lab[-1,:].tolist()) \
           | set(lab[:,0].tolist()) | set(lab[:,-1].tolist())
    border.discard(0)
    bg = np.isin(lab, list(border))
    char = ~bg
    char = ndimage.binary_closing(char, iterations=2)
    char = ndimage.binary_opening(char, iterations=1)
    char = ndimage.binary_fill_holes(char)
else:
    char = ~light

print("char: %d px (%.2f%%)" % (char.sum(), 100.0*char.sum()/char.size))

# ============================================================
# B) 6 depth layers within char
# ============================================================
char_depth = d[char]
qs = np.percentile(char_depth, [100/6, 200/6, 300/6, 400/6, 500/6])
print("\nchar depth quintiles:", ["%.1f"%q for q in qs])

colors = [
    (230, 60, 60),    # L1 nearest
    (240, 150, 60),   # L2
    (240, 220, 60),   # L3
    (100, 220, 100),  # L4
    (80, 160, 240),   # L5
    (140, 80, 220),   # L6 farthest
]

masks = []
m0 = char & (d < qs[0]); masks.append(m0)
for i in range(1, 5):
    masks.append(char & (d >= qs[i-1]) & (d < qs[i]))
masks.append(char & (d >= qs[4]))

vis = rgb.copy()
for i, m in enumerate(masks):
    print("  L%d: %8d px (%.1f%% of char)" % (i+1, m.sum(), 100.0*m.sum()/char.sum()))
    c = np.array(colors[i])
    vis[m] = (0.6 * vis[m] + 0.4 * c).astype(np.float32)

Image.fromarray(vis.astype(np.uint8)).save(f"{ROOT}/diag_layers6.png")

vis_only = np.zeros_like(rgb)
for i, m in enumerate(masks):
    vis_only[m] = colors[i]
Image.fromarray(vis_only.astype(np.uint8)).save(f"{ROOT}/diag_layers6_only.png")
print("saved diag_layers6.png / diag_layers6_only.png")

# ============================================================
# C) FACE / HAIR by depth + position only (no color)
# ============================================================
print("\n[face/hair by depth+position]")
top_y = int(0.42 * H)

# face: char ∩ depth 상위 ∩ central-upper zone
fy0, fy1 = int(0.18*H), int(0.50*H)
fx0, fx1 = int(0.15*W), int(0.55*W)
face_zone = np.zeros_like(char); face_zone[fy0:fy1, fx0:fx1] = True
d_thr = np.percentile(d[char], 60)
face = char & face_zone & (d >= d_thr)

if HAS_SCIPY:
    face = ndimage.binary_closing(face, iterations=3)
    face = ndimage.binary_fill_holes(face)
    lab2, n2 = ndimage.label(face)
    if n2 > 0:
        sizes = ndimage.sum(face, lab2, range(1, n2+1))
        big = np.argmax(sizes) + 1
        face = (lab2 == big)
print("face: %d px (%.2f%%)" % (face.sum(), 100.0*face.sum()/face.size))

# hair: top-42% ∩ char ∩ ¬face
hair_zone = np.zeros_like(char); hair_zone[:top_y, :] = True
hair = char & hair_zone & (~face)
print("hair(region): %d px" % hair.sum())

hair_d = d[hair]
if len(hair_d) > 0:
    med = np.median(hair_d)
    print("hair median depth: %.1f" % med)
    hair_front = hair & (d >= med)
    hair_back  = hair & (d <  med)
    print("  hair_front: %d  hair_back: %d" % (hair_front.sum(), hair_back.sum()))
else:
    hair_front = np.zeros_like(char); hair_back = np.zeros_like(char)

vis3 = rgb.copy()
vis3[hair_back]  = (0.45*vis3[hair_back]  + 0.55*np.array([60,60,255])).astype(np.float32)
vis3[hair_front] = (0.45*vis3[hair_front] + 0.55*np.array([255,60,60])).astype(np.float32)
vis3[face]       = (0.45*vis3[face]       + 0.55*np.array([60,255,60])).astype(np.float32)
Image.fromarray(vis3.astype(np.uint8)).save(f"{ROOT}/diag_fch.png")
print("saved diag_fch.png (green=face, red=hair_front, blue=hair_back)")
