
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGFORMER_REPO = PROJECT_ROOT / "external" / "SegFormer3D"

if str(SEGFORMER_REPO) not in sys.path:
    sys.path.insert(0, str(SEGFORMER_REPO))

from architectures.segformer3d import build_segformer3d_model


def build_model():
    config = {
        "model_name": "segformer3d",
        "model_parameters": {
            "in_channels": 1,
            "sr_ratios": [4, 2, 1, 1],
            "embed_dims": [16, 32, 64, 128],
            "patch_kernel_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "mlp_ratios": [4, 4, 4, 4],
            "num_heads": [1, 2, 4, 8],
            "depths": [2, 2, 2, 2],
            "decoder_head_embedding_dim": 128,
            "num_classes": 2,
            "decoder_dropout": 0.0,
        }
    }

    return build_segformer3d_model(config)


if __name__ == "__main__":
    torch.set_num_threads(6)

    model = build_model()
    model.eval()

    x = torch.randn(1, 1, 64, 64, 32)

    with torch.no_grad():
        output = model(x)

    parameters = sum(p.numel() for p in model.parameters())

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Finite output:", torch.isfinite(output).all().item())
    print("Parameters:", f"{parameters:,}")
