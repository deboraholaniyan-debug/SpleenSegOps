
from pathlib import Path
import torch
import pandas as pd

from config import PROCESSED_DATA_DIR, SPLIT_MANIFEST

manifest = pd.read_csv(SPLIT_MANIFEST)
labelled = manifest[
    manifest["split"].isin(["train", "validation"])
]

errors = []
summary = []

for _, row in labelled.iterrows():
    patient_id = row["patient_id"]
    file_path = PROCESSED_DATA_DIR / f"{patient_id}.pt"

    if not file_path.exists():
        errors.append(f"Missing file: {patient_id}")
        continue

    data = torch.load(file_path, map_location="cpu", weights_only=False)

    image = data["image"]
    label = data["label"]

    if image.shape != label.shape:
        errors.append(f"Shape mismatch: {patient_id}")

    if not torch.isfinite(image).all():
        errors.append(f"Non-finite image values: {patient_id}")

    if not torch.isin(label, torch.tensor([0, 1])).all():
        errors.append(f"Unexpected mask labels: {patient_id}")

    if label.sum() == 0:
        errors.append(f"Empty spleen mask: {patient_id}")

    summary.append({
        "patient_id": patient_id,
        "split": row["split"],
        "shape": tuple(image.shape),
        "image_min": image.min().item(),
        "image_max": image.max().item(),
        "spleen_voxels": int(label.sum().item()),
    })

print("Processed cases checked:", len(summary))
print("Validation errors:", len(errors))

for error in errors:
    print("ERROR:", error)

if not errors:
    print("\nAll processed cases passed validation.")

print("\nSample records:")
for record in summary[:5]:
    print(record)
