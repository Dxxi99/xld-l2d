import numpy as np
from PIL import Image
import os

OUT   = os.path.expanduser("~/XLD-L2D/assets/real/depth_anything_v2.png")
IMG   = os.path.expanduser("~/XLD-L2D/assets/real/my_vtuber.png")

depth = np.array(Image.open(OUT).convert("L")).astype(np.float32)
rgb   = np.array(Image.open(IMG).convert("RGB")).astype(np.float32)

# 피부 마스크 (간단 휴리스틱)
R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
skin = (R > 150) & (R > G) & (R > B) & (abs(R-G) < 60) & (R-B > 15)

# face bbox 내부만
fy0, fy1, fx0, fx1 = 539, 1157, 234, 919
region = skin[fy0:fy1, fx0:fx1]
depth_region = depth[fy0:fy1, fx0:fx1]

if region.sum() == 0:
    print("no skin pixels found in bbox")
else:
    d_skin = depth_region[region]
    print("skin pixels: %d" % region.sum())
    print("skin depth: min %.1f max %.1f mean %.1f std %.1f" % (
        d_skin.min(), d_skin.max(), d_skin.mean(), d_skin.std()))
    print("bbox depth (with bg): std %.1f" % depth_region.std())
    print("ratio: skin std / bbox std = %.3f" % (d_skin.std() / depth_region.std()))
