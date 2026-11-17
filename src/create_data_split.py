
from pathlib import Path
import json

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
VALIDATION_SIZE = 8

# Read the official dataset metadata
with open(RAW_DIR / "dataset.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

# Calculate physical spleen volume for each labelled patient
records = []

for item in metadata["training"]:
    image_relative = item["image"].replace("./", "")
    label_relative = item["label"].replace("./", "")

    image_path = RAW_DIR / image_relative
    label_path = RAW_DIR / label_relative

    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Missing image or label for {image_relative}"
        )

    image = nib.load(str(image_path))
    label = nib.load(str(label_path))

    mask = np.asanyarray(label.dataobj)

    if not np.isin(np.unique(mask), [0, 1]).all():
        raise ValueError(f"Unexpected labels in {label_relative}")

    if image.shape != label.shape:
        raise ValueError(f"Image/mask shape mismatch: {image_relative}")

    voxel_volume_mm3 = float(np.prod(label.header.get_zooms()[:3]))
    spleen_voxels = int(np.count_nonzero(mask == 1))
    spleen_volume_mm3 = spleen_voxels * voxel_volume_mm3

    records.append({
        "patient_id": Path(image_relative).stem,
        "image": image_relative,
        "label": label_relative,
        "spleen_voxels": spleen_voxels,
        "spleen_volume_mm3": spleen_volume_mm3
    })

df = pd.DataFrame(records)

if len(df) != 41:
    raise ValueError(f"Expected 41 labelled patients, found {len(df)}")

# Divide patients into four spleen-volume groups
df["volume_group"] = pd.qcut(
    df["spleen_volume_mm3"],
    q=4,
    labels=False,
    duplicates="drop"
)

# Create a reproducible patient-level split
train_df, val_df = train_test_split(
    df,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["volume_group"]
)

train_df = train_df.copy()
val_df = val_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"

# Add the official test cases without labels
test_records = []

for image_relative in metadata["test"]:
    image_relative = image_relative.replace("./", "")

    if not (RAW_DIR / image_relative).exists():
        raise FileNotFoundError(f"Missing test image: {image_relative}")

    test_records.append({
        "patient_id": Path(image_relative).stem,
        "image": image_relative,
        "label": None,
        "spleen_voxels": None,
        "spleen_volume_mm3": None,
        "volume_group": None,
        "split": "official_test"
    })

test_df = pd.DataFrame(test_records)

manifest = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

manifest.to_csv(
    OUTPUT_DIR / "split_manifest.csv",
    index=False
)

print("Dataset split completed.")
print(manifest["split"].value_counts())
print("\nTraining and validation spleen-volume summary:")
print(
    manifest[manifest["split"] != "official_test"]
    .groupby("split")["spleen_volume_mm3"]
    .describe()
)
print("\nSaved:", OUTPUT_DIR / "split_manifest.csv")
