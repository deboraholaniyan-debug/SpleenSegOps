from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset
SPLIT_MANIFEST = PROCESSED_DATA_DIR / "split_manifest.csv"
IMAGE_SIZE = (512, 512)

# Preprocessing
TARGET_SPACING = (1.0, 1.0, 2.5)
CT_INTENSITY_MIN = -125.0
CT_INTENSITY_MAX = 275.0

# Training
RANDOM_SEED = 42
NUM_CLASSES = 2
IN_CHANNELS = 1
PATCH_SIZE = (64, 64, 32)
BATCH_SIZE = 1
NUM_WORKERS = 0

# CPU execution
DEVICE = "cpu"