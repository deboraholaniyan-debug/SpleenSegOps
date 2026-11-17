
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    SPLIT_MANIFEST,
    MODELS_DIR,
    PATCH_SIZE,
    DEVICE,
)

from slimformer3d_model import build_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHECKPOINT_PATH = MODELS_DIR / "slimformer3d_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAILED_OUTPUT = OUTPUT_DIR / "slimformer3d_detailed_metrics.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "slimformer3d_evaluation_summary.csv"

torch.set_num_threads(6)
device = torch.device(DEVICE)


# ============================================================
# 2. LOAD VALIDATION PATIENTS
# ============================================================

split_df = pd.read_csv(SPLIT_MANIFEST)

validation_patients = split_df.loc[
    split_df["split"] == "validation",
    "patient_id",
].tolist()

print("Validation patients:", len(validation_patients))


# ============================================================
# 3. LOAD BEST SEGFORMER3D CHECKPOINT
# ============================================================

model = build_model().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print(
    "Best training-run validation Dice:",
    checkpoint["best_validation_dice"],
)


# ============================================================
# 4. POST-PROCESSING FUNCTIONS
# ============================================================

def largest_connected_component(mask):
    """
    Retain the largest 3D foreground connected component.
    """
    mask = mask.astype(bool)

    labeled_mask, number_of_components = ndimage.label(mask)

    if number_of_components == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0

    largest_component = component_sizes.argmax()

    return labeled_mask == largest_component


def apply_postprocessing(prediction, method):
    """
    Apply one of the three evaluation configurations.
    """
    prediction = prediction.astype(bool)

    if method == "Original":
        return prediction

    if method == "Largest_Component":
        return largest_connected_component(prediction)

    if method == "Largest_Component_Hole_Filling":
        prediction = largest_connected_component(prediction)
        prediction = ndimage.binary_fill_holes(prediction)
        return prediction.astype(bool)

    raise ValueError(f"Unknown post-processing method: {method}")


# ============================================================
# 5. METRIC CALCULATION
# ============================================================

def calculate_metrics(prediction, ground_truth):
    """
    Calculate binary segmentation metrics for one patient.
    """
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()
    tn = np.logical_and(~prediction, ~ground_truth).sum()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    iou = tp / (tp + fp + fn + 1e-8)

    precision = tp / (tp + fp + 1e-8)

    recall = tp / (tp + fn + 1e-8)

    specificity = tn / (tn + fp + 1e-8)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    f1_score = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1_score,
        "Dice": dice,
        "IoU": iou,
        "Specificity": specificity,
    }


# ============================================================
# 6. FULL-VOLUME VALIDATION
# ============================================================

methods = [
    "Original",
    "Largest_Component",
    "Largest_Component_Hole_Filling",
]

results = []

print("\nStarting Slimformer3d post-processing evaluation...\n")

with torch.no_grad():

    for patient_id in tqdm(
        validation_patients,
        desc="Evaluating validation patients",
    ):

        file_path = PROCESSED_DATA_DIR / f"{patient_id}.pt"

        data = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )

        image = (
            data["image"]
            .float()
            .unsqueeze(0)
            .to(device)
        )

        label = (
            data["label"]
            .long()
            .unsqueeze(0)
            .to(device)
        )

        # Full-volume sliding-window inference
        logits = sliding_window_inference(
            inputs=image,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

        prediction = torch.argmax(
            logits,
            dim=1,
        )

        prediction = (
            prediction[0]
            .cpu()
            .numpy() == 1
        )

        ground_truth = (
            label[0, 0]
            .cpu()
            .numpy() == 1
        )

        # Evaluate all three post-processing configurations
        for method in methods:

            processed_prediction = apply_postprocessing(
                prediction,
                method,
            )

            metrics = calculate_metrics(
                processed_prediction,
                ground_truth,
            )

            results.append({
                "Patient_ID": patient_id,
                "Method": method,
                **metrics,
            })

        del data, image, label, logits, prediction, ground_truth

        gc.collect()


# ============================================================
# 7. SAVE PATIENT-LEVEL METRICS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    DETAILED_OUTPUT,
    index=False,
)


# ============================================================
# 8. CALCULATE MEAN VALIDATION METRICS
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "Dice",
    "IoU",
    "Specificity",
]

summary_df = (
    results_df
    .groupby("Method")[metric_columns]
    .mean()
    .reset_index()
)

summary_df = summary_df.sort_values(
    by="Dice",
    ascending=False,
)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\nPATIENT-LEVEL RESULTS")
print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nMEAN VALIDATION RESULTS")
print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nEvaluation completed.")
print("Detailed metrics:", DETAILED_OUTPUT)
print("Summary metrics:", SUMMARY_OUTPUT)
