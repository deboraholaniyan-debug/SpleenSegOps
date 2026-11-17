
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIMFORMER_REPO = PROJECT_ROOT / "external" / "SlimFormer3D"

if str(SLIMFORMER_REPO) not in sys.path:
    sys.path.insert(0, str(SLIMFORMER_REPO))

from lib.models.SlimFormer_base import SlimFormer
from lib.models.mymodules.former import ClassAttention


def build_model():
    return SlimFormer(
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
