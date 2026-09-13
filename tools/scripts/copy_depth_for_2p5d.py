#!/usr/bin/env python3
"""copy_depth_for_2p5d.py — depth.png를 다운샘플해서 model dir에 저장."""
import sys
from pathlib import Path
from PIL import Image

src = Path(sys.argv[1]).expanduser().resolve()
dst_dir = Path(sys.argv[2]).expanduser().resolve()
dst_dir.mkdir(parents=True, exist_ok=True)

img = Image.open(src).convert("L")
W, H = img.size
scale = min(1024.0 / W, 1024.0 / H, 1.0)
if scale < 1.0:
    img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
out = dst_dir / "depth_small.png"
img.save(out)
print(f"[done] {out}  {img.size}")
