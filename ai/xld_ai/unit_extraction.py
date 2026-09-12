"""Per-pixel unit extraction.

Atomic Unit = 1 pixel.
Inference Unit = context block (tile).
Decision Unit = 1 pixel.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PixelUnit:
    x: int
    y: int


@dataclass
class ContextBlock:
    origin: PixelUnit
    width: int
    height: int
    rgba: list  # length = width * height * 4


def split_into_context_blocks(w: int, h: int, tile: int = 16, overlap: int = 4) -> list[ContextBlock]:
    blocks: list[ContextBlock] = []
    step = max(1, tile - overlap)
    for y in range(0, h, step):
        for x in range(0, w, step):
            tw = min(tile, w - x)
            th = min(tile, h - y)
            blocks.append(ContextBlock(PixelUnit(x, y), tw, th, []))
    return blocks
