
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

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
scores = {threshold: [] for threshold in thresholds}

with torch.no_grad():
    for _, row in cases.iterrows():
        patient_id = row["patient_id"]

        data = torch.load(
            PROCESSED_DATA_DIR / f"{patient_id}.pt",
            map_location="cpu",
            weights_only=False,
        )

        image = data["image"].unsqueeze(0).to(device)
        target = data["label"][0].to(device)

        logits = sliding_window_inference(
            image,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

        probabilities = torch.softmax(logits, dim=1)[0, 1]

        for threshold in thresholds:
            prediction = probabilities >= threshold

            intersection = (
                prediction & (target == 1)
            ).sum().float()

            denominator = (
                prediction.sum() + (target == 1).sum()
            ).float()

            dice = (
                (2 * intersection + 1e-6)
                / (denominator + 1e-6)
            ).item()

            scores[threshold].append(dice)

for threshold, values in scores.items():
    print(
        f"Threshold={threshold:.1f} | "
        f"Mean Dice={sum(values) / len(values):.4f}"
    )
