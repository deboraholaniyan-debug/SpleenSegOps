
import torch
import pandas as pd
from monai.inferers import sliding_window_inference

from config import (
    PROCESSED_DATA_DIR,
    SPLIT_MANIFEST,
    MODELS_DIR,
    PATCH_SIZE,
    DEVICE,
)
from model import build_model

device = torch.device(DEVICE)

checkpoint = torch.load(
    MODELS_DIR / "baseline_3d_unet_best.pth",
    map_location=device,
    weights_only=False,
)

model = build_model().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

manifest = pd.read_csv(SPLIT_MANIFEST)
cases = manifest[manifest["split"] == "validation"]

with torch.no_grad():
    for _, row in cases.iterrows():
        patient_id = row["patient_id"]

        data = torch.load(
            PROCESSED_DATA_DIR / f"{patient_id}.pt",
            map_location="cpu",
            weights_only=False,
        )

        image = data["image"].unsqueeze(0).to(device)
        label = data["label"].to(device)

        logits = sliding_window_inference(
            image,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

        prediction = torch.argmax(logits, dim=1)[0]
        target = label[0]

        predicted_voxels = (prediction == 1).sum().item()
        actual_voxels = (target == 1).sum().item()

        intersection = (
            (prediction == 1) & (target == 1)
        ).sum().item()

        print(
            f"{patient_id}: "
            f"Predicted={predicted_voxels:,}, "
            f"Actual={actual_voxels:,}, "
            f"Intersection={intersection:,}"
        )

