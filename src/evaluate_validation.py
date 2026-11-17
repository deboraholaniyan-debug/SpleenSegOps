
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from monai.inferers import sliding_window_inference

from config import (
    PROCESSED_DATA_DIR,
    SPLIT_MANIFEST,
    MODELS_DIR,
    PATCH_SIZE,
    DEVICE,
)
from model import build_model


# ==========================================
# 1. LOAD CHECKPOINT AND VALIDATION PATIENTS
# ==========================================

device = torch.device(DEVICE)

checkpoint_path = MODELS_DIR / "baseline_3d_unet_best.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

model = build_model().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

manifest = pd.read_csv(SPLIT_MANIFEST)
validation_cases = manifest[
    manifest["split"] == "validation"
]["patient_id"].tolist()

print("Checkpoint:", checkpoint_path)
print("Validation patients:", len(validation_cases))


# ==========================================
# 2. METRIC CALCULATION
# ==========================================

def calculate_metrics(prediction, ground_truth):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction, ground_truth
    ).sum()

    pred_volume = prediction.sum()
    true_volume = ground_truth.sum()

    dice = (
        2.0 * intersection /
        (pred_volume + true_volume + 1e-8)
    )

    iou = (
        intersection /
        (pred_volume + true_volume - intersection + 1e-8)
    )

    sensitivity = (
        intersection / (true_volume + 1e-8)
    )

    return {
        "Dice": dice,
        "IoU": iou,
        "Sensitivity": sensitivity,
        "Predicted_Voxels": int(pred_volume),
        "GroundTruth_Voxels": int(true_volume),
    }


# ==========================================
# 3. FULL-VOLUME VALIDATION
# ==========================================

results = []

for patient_id in tqdm(validation_cases):

    data = torch.load(
        PROCESSED_DATA_DIR / f"{patient_id}.pt",
        map_location="cpu",
        weights_only=False,
    )

    image = data["image"].unsqueeze(0).to(device)
    ground_truth = (
        data["label"].squeeze(0).numpy() > 0
    )

    with torch.no_grad():

        logits = sliding_window_inference(
            inputs=image,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

        prediction = (
            torch.argmax(logits, dim=1)[0]
            .cpu()
            .numpy()
        )

    metrics = calculate_metrics(
        prediction,
        ground_truth,
    )

    metrics["Patient_ID"] = patient_id
    results.append(metrics)

    print(
        f"\n{patient_id}: "
        f"Dice={metrics['Dice']:.4f}, "
        f"IoU={metrics['IoU']:.4f}, "
        f"Sensitivity={metrics['Sensitivity']:.4f}"
    )


# ==========================================
# 4. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

output_path = (
    PROCESSED_DATA_DIR.parent.parent
    / "reports"
    / "validation_metrics.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

results_df.to_csv(output_path, index=False)

print("\n========== VALIDATION SUMMARY ==========")

print(
    results_df[
        ["Dice", "IoU", "Sensitivity"]
    ].mean().round(4)
)

print("\nPatient-level results:")
print(results_df.to_string(index=False))

print("\nResults saved to:", output_path)
