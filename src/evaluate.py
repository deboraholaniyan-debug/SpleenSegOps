
import pandas as pd
import torch
from monai.inferers import sliding_window_inference

from config import (
    PROCESSED_DATA_DIR,
    SPLIT_MANIFEST,
    MODELS_DIR,
    PATCH_SIZE,
    DEVICE,
)
from model import build_model


def calculate_dice(prediction, target):
    intersection = ((prediction == 1) & (target == 1)).sum().float()
    denominator = (prediction == 1).sum() + (target == 1).sum()

    return ((2 * intersection + 1e-6) / (denominator + 1e-6)).item()


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
validation_cases = manifest[
    manifest["split"] == "validation"
]

results = []

with torch.no_grad():
    for _, row in validation_cases.iterrows():
        patient_id = row["patient_id"]

        data = torch.load(
            PROCESSED_DATA_DIR / f"{patient_id}.pt",
            map_location="cpu",
            weights_only=False,
        )

        image = data["image"].unsqueeze(0).to(device)
        label = data["label"].to(device)

        logits = sliding_window_inference(
            inputs=image,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

        prediction = torch.argmax(logits, dim=1)[0]
        target = label[0]

        dice = calculate_dice(prediction, target)

        results.append({
            "patient_id": patient_id,
            "dice": dice,
        })

        print(f"{patient_id}: Dice = {dice:.4f}")

results_df = pd.DataFrame(results)

output_path = MODELS_DIR / "baseline_validation_results.csv"
results_df.to_csv(output_path, index=False)

print("\nFull-volume validation results")
print(results_df.to_string(index=False))
print(f"\nMean Dice: {results_df['dice'].mean():.4f}")
print(f"Standard deviation: {results_df['dice'].std():.4f}")
print("Results saved to:", output_path)
