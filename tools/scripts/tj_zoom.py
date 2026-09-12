import numpy as np
from PIL import Image
import os
ROOT = os.path.expanduser("~/XLD-L2D/assets/real")

vis = np.array(Image.open(f"{ROOT}/pixel_tjunction_v3.png"))

# 얼굴 왼쪽 눈 주변 (T-junction 밀집 예상)
regions = [
    ("eye",   (620,  720,  880,  920)),   # 왼쪽 눈
    ("bangs", (500,  550,  900,  850)),   # 앞머리-얼굴 경계
    ("mouth", (700,  900,  950, 1080)),   # 입
]
for name, (x0,y0,x1,y1) in regions:
    crop = vis[y0:y1, x0:x1]
    big = Image.fromarray(crop).resize(
        (crop.shape[1]*3, crop.shape[0]*3), Image.NEAREST)
    big.save(f"{ROOT}/tj_zoom_{name}.png")
    print(f"saved tj_zoom_{name}.png  ({x1-x0}x{y1-y0} -> 3x)")
