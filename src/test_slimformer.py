import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external" / "SlimFormer3D"

sys.path.insert(0, str(REPO))

from lib.models.SlimFormer_base import SlimFormer
from lib.models.mymodules.former import ClassAttention

torch.set_num_threads(6)

model = SlimFormer(
    in_channels=1,
    out_channels=2,
    patch_size=2,
    base_channels=16,
    num_heads=(2, 4, 8, 16),
    token_mixer=ClassAttention,
    drop=0.0,
    drop_path=0.0,
    metaformer_layers=(1, 1, 1, 1),
)

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

model.eval()
x = torch.randn(1, 1, 64, 64, 32)

with torch.no_grad():
    y = model(x)

print("Input:", tuple(x.shape))
print("Output:", tuple(y.shape))
print("Parameters:", f"{parameters:,}")
print("Finite output:", torch.isfinite(y).all().item())
