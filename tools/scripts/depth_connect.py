import numpy as np
from PIL import Image
import os

ROOT = os.path.expanduser("~/XLD-L2D/assets/real")
img = Image.open(f"{ROOT}/my_vtuber.png").convert("RGB")
depth = Image.open(f"{ROOT}/depth_anything_v2.png").convert("L")

rgb = np.array(img).astype(np.float32)
d   = np.array(depth).astype(np.float32)
print("img", img.size, "depth", depth.size)

# ============================================================
# 1) FACE REFINEMENT
# ============================================================
fx0, fy0, fx1, fy1 = 234, 539, 919, 1157
face_rgb   = rgb[fy0:fy1, fx0:fx1]
face_depth = d[fy0:fy1, fx0:fx1]

hist, edges = np.histogram(face_depth, bins=16, range=(0, 255))
print("\n[face box depth histogram]")
for i in range(16):
    bar = "#" * int(hist[i] / max(1, hist.max()) * 40)
    print("  %3d-%3d: %6d %s" % (edges[i], edges[i+1], hist[i], bar))

R, G, B = face_rgb[...,0], face_rgb[...,1], face_rgb[...,2]
skin = (R > 150) & (R > G) & (R > B) & (np.abs(R-G) < 60) & (R-B > 15)
print("\nskin pixels in face box:", skin.sum())

if skin.sum() > 0:
    sd = face_depth[skin]
    print("skin depth percentiles:")
    for p in [5, 25, 50, 75, 95]:
        print("  %3d%%: %.1f" % (p, np.percentile(sd, p)))
    lo, hi = np.percentile(sd, [10, 90])
    print("keep range: [%.1f, %.1f]" % (lo, hi))

    refined = skin & (face_depth >= lo) & (face_depth <= hi)
    print("refined: %d / %d (%.1f%%)" % (refined.sum(), skin.sum(), 100.0*refined.sum()/skin.sum()))
    print("refined std: %.1f (was %.1f)" % (face_depth[refined].std(), sd.std()))

    # visualize
    vis = rgb.astype(np.uint8).copy()
    vis[fy0:fy0+3, fx0:fx1] = [255, 255, 0]
    vis[fy1-3:fy1, fx0:fx1] = [255, 255, 0]
    vis[fy0:fy1, fx0:fx0+3] = [255, 255, 0]
    vis[fy0:fy1, fx1-3:fx1] = [255, 255, 0]
    reg = vis[fy0:fy1, fx0:fx1].astype(np.float32)
    m = np.zeros_like(reg)
    m[refined] = [255, 60, 60]
    vis[fy0:fy1, fx0:fx1] = ((1-0.4)*reg + 0.4*m).astype(np.uint8)
    Image.fromarray(vis).save(f"{ROOT}/face_refined.png")
    print("saved face_refined.png")

# ============================================================
# 2) HAIR FRONT/BACK SPLIT
# ============================================================
R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
mx = np.maximum(np.maximum(R, G), B)
mn = np.minimum(np.minimum(R, G), B)
sat = (mx - mn) / (mx + 1e-6)
val = mx / 255.0
hair = (B > R + 20) & (sat > 0.2) & (val > 0.3)
print("\n[hair detection]")
print("hair pixels (blue/purple):", hair.sum())

top_y = int(0.42 * rgb.shape[0])
hair_top = hair.copy()
hair_top[top_y:, :] = False
print("hair in top 42%%: %d (%.2f%%)" % (hair_top.sum(), 100.0*hair_top.sum()/hair_top.size))

hair_depth_vals = d[hair_top]
if len(hair_depth_vals) > 0:
    med = np.median(hair_depth_vals)
    print("hair depth: min %.1f max %.1f med %.1f" % (
        hair_depth_vals.min(), hair_depth_vals.max(), med))

    front = hair_top & (d >= med)
    back  = hair_top & (d <  med)
    print("front: %d back: %d" % (front.sum(), back.sum()))

    vis2 = rgb.astype(np.uint8).copy()
    vis2[back]  = [60, 60, 255]
    vis2[front] = [255, 60, 60]
    Image.fromarray(vis2).save(f"{ROOT}/hair_split.png")
    print("saved hair_split.png")

    # hair depth histogram
    hist, edges = np.histogram(hair_depth_vals, bins=16, range=(0, 255))
    print("\n[hair depth histogram]")
    for i in range(16):
        bar = "#" * int(hist[i] / max(1, hist.max()) * 40)
        print("  %3d-%3d: %6d %s" % (edges[i], edges[i+1], hist[i], bar))
