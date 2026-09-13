#!/usr/bin/env python3
"""convert_esrgan_to_coreml.py — Real-ESRGAN x4plus_anime_6B (PyTorch) → CoreML

1회성 변환 스크립트. 변환 후에는 torch 불필요.

Usage:
    python3 convert_esrgan_to_coreml.py <weights.pth> <out.mlpackage>
"""
import sys
import torch
import numpy as np
import torch.nn as nn
import coremltools as ct
from pathlib import Path

# ─────────────────────────────────────────────────────────
# RRDBNet (Real-ESRGAN x4plus_anime_6B 전용 아키텍처)
# 원본: https://github.com/xinntao/Real-ESRGAN
# ─────────────────────────────────────────────────────────
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ─────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    pth = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]).expanduser().resolve()

    print(f"[load] {pth}")
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32)

    state = torch.load(str(pth), map_location="cpu", weights_only=True)
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    net.load_state_dict(state, strict=True)
    net.eval()
    print("[ok] weights loaded")

    # trace with fixed input (256x256 → 1024x1024)
    dummy = torch.randn(1, 3, 512, 512)
    print("[trace] dummy", tuple(dummy.shape))
    with torch.no_grad():
        traced = torch.jit.trace(net, dummy)

    print("[convert] to CoreML ...")
    # 가변 크기: MultiArrayType + EnumeratedShapes
    # ImageType은 width/height 고정이라 가변 불가 → TensorType 사용
    shapes = [(1, 3, 256, 256), (1, 3, 384, 384), (1, 3, 512, 512), (1, 3, 640, 640), (1, 3, 768, 768)]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input",
            shape=ct.EnumeratedShapes(shapes=shapes, default=(1, 3, 512, 512)),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.short_description = "Real-ESRGAN x4plus anime 6B"
    mlmodel.input_description["input"] = "LR image (any size, multiple of 4 recommended)"
    mlmodel.output_description["output"] = "4x upscaled image"
    mlmodel.save(str(out))
    print(f"[done] saved: {out}")


if __name__ == "__main__":
    main()
