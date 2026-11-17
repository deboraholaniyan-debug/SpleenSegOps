
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import ndimage
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
# 1. LOAD MODEL AND VALIDATION CASES
# ==========================================

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

validation_cases = manifest.loc[
    manifest["split"] == "validation",
    "patient_id"
].tolist()

# ==========================================
# 2. CALCULATE EVALUATION METRICS
# ==========================================

def calculate_metrics(prediction, ground_truth):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(prediction, ground_truth).sum()
    tn = np.logical_and(~prediction, ~ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    precision = tp / (tp + fp + 1e-8)

    recall = tp / (tp + fn + 1e-8)

    f1_score = (
        2 * precision * recall /
        (precision + recall + 1e-8)
    )

    dice = (
        2 * tp /
        (2 * tp + fp + fn + 1e-8)
    )

    iou = (
        tp /
        (tp + fp + fn + 1e-8)
    )

    specificity = tn / (tn + fp + 1e-8)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1_score,
        "Dice": dice,
        "IoU": iou,
        "Specificity": specificity,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

# ==========================================
# 3. POST-PROCESSING METHODS
# ==========================================

def largest_connected_component(mask):

    components, number = ndimage.label(mask)

    if number == 0:
        return mask.astype(bool)

    sizes = np.bincount(components.ravel())
    sizes[0] = 0

    return components == np.argmax(sizes)

def fill_holes(mask):

    return ndimage.binary_fill_holes(mask)

# ==========================================
# 4. EVALUATE ALL VALIDATION PATIENTS
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
            .numpy() > 0
        )

    largest = largest_connected_component(prediction)

    largest_filled = fill_holes(largest)

    methods = {
        "Original_3D_UNet": prediction,
        "3D_UNet_Largest_Component": largest,
        "3D_UNet_LCC_Hole_Filling": largest_filled,
    }

    for method_name, processed_mask in methods.items():

        metrics = calculate_metrics(
            processed_mask,
            ground_truth,
        )

        metrics["Patient_ID"] = patient_id
        metrics["Method"] = method_name

        results.append(metrics)

# ==========================================
# 5. SAVE PATIENT-LEVEL RESULTS
# ==========================================

results_df = pd.DataFrame(results)

reports_dir = (
    PROCESSED_DATA_DIR.parent.parent / "reports"
)

reports_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    reports_dir / "baseline_detailed_metrics.csv",
    index=False,
)

# ==========================================
# 6. CALCULATE MEAN VALIDATION SCORES
# ==========================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "Dice",
    "IoU",
    "Specificity",
]

summary = (
    results_df.groupby("Method")[metric_columns]
    .mean()
    .round(4)
)

summary.to_csv(
    reports_dir / "baseline_evaluation_summary.csv"
)

print("\n========== BASELINE EVALUATION SUMMARY ==========")

print(summary.to_string())

print("\n========== PATIENT-LEVEL RESULTS ==========")

print(
    results_df[
        ["Patient_ID", "Method"] + metric_columns
    ].to_string(index=False)
)

print("\nResults saved to:")
print(reports_dir / "baseline_evaluation_summary.csv")
print(reports_dir / "baseline_detailed_metrics.csv")