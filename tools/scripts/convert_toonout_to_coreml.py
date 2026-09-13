#!/usr/bin/env python3
"""convert_toonout_to_coreml.py — ToonOut (BiRefNet fine-tune) .pth → CoreML

Usage:
    python3 convert_toonout_to_coreml.py <model.pth> <out.mlpackage>
"""
import sys
from pathlib import Path
import numpy as np
import torch
import coremltools as ct

# BiRefNet 소스 경로
BIREFNET_SRC = Path(__file__).resolve().parent.parent.parent / "models/toonout/BiRefNet-src"
sys.path.insert(0, str(BIREFNET_SRC))


class BiRefNetWrapper(torch.nn.Module):
    """BiRefNet을 감싸서 최종 마스크만 반환 (trace용)"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        # out[3] = 최종 마스크 (1,1,H,W)
        return out[3]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    pth_path = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path(sys.argv[2]).expanduser().resolve()

    from birefnet.models.birefnet import BiRefNet

    print(f"[load] BiRefNet ...")
    model = BiRefNet(bb_pretrained=False)
    sd = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    new_sd = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v
              for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[load] missing={len(missing)}  unexpected={len(unexpected)}")
    model.eval()

    wrapper = BiRefNetWrapper(model)
    wrapper.eval()

    # trace (1024)
    dummy = torch.randn(1, 3, 1024, 1024)
    print("[trace] ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    # CoreML 변환 (가변 크기)
    shapes = [(1, 3, 512, 512), (1, 3, 768, 768), (1, 3, 1024, 1024), (1, 3, 1536, 1536)]
    print("[convert] to CoreML ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="input",
            shape=ct.EnumeratedShapes(shapes=shapes, default=(1, 3, 1024, 1024)),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="mask", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlmodel.short_description = "ToonOut (BiRefNet anime fine-tune)"
    mlmodel.input_description["input"] = "RGB normalized (ImageNet), (1,3,H,W)"
    mlmodel.output_description["mask"] = "foreground mask (1,1,H,W) 0..1"
    mlmodel.save(str(out_path))
    print(f"[done] saved: {out_path}")


if __name__ == "__main__":
    main()
