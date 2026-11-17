
from pathlib import Path
import gc
import random

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

from model import build_model as build_unet
from segformer3d_model import build_model as build_segformer
from slimformer3d_model import build_model as build_slimformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAILED_OUTPUT = OUTPUT_DIR / "all_models_surface_metrics.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "all_models_surface_summary.csv"

CHECKPOINTS = {
    "3D_UNet": MODELS_DIR / "baseline_3d_unet_best.pth",
    "SegFormer3D": MODELS_DIR / "segformer3d_best.pth",
    "SlimFormer3D": MODELS_DIR / "slimformer3d_best.pth",
}

MODEL_BUILDERS = {
    "3D_UNet": build_unet,
    "SegFormer3D": build_segformer,
    "SlimFormer3D": build_slimformer,
}

METHODS = [
    "Original",
    "Largest_Component",
    "Largest_Component_Hole_Filling",
]

torch.set_num_threads(6)
device = torch.device(DEVICE)


# ============================================================
# 2. VALIDATION PATIENTS
# ============================================================

split_df = pd.read_csv(SPLIT_MANIFEST)

validation_patients = split_df.loc[
    split_df["split"] == "validation",
    "patient_id",
].tolist()

print("Device:", device)
print("Validation patients:", len(validation_patients))


# ============================================================
# 3. POST-PROCESSING
# ============================================================

def largest_connected_component(mask):
    mask = mask.astype(bool)

    labeled, number = ndimage.label(mask)

    if number == 0:
        return np.zeros_like(mask, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    return labeled == sizes.argmax()


def apply_postprocessing(prediction, method):

    prediction = prediction.astype(bool)

    if method == "Original":
        return prediction

    if method == "Largest_Component":
        return largest_connected_component(prediction)

    if method == "Largest_Component_Hole_Filling":
        prediction = largest_connected_component(prediction)
        return ndimage.binary_fill_holes(prediction)

    raise ValueError(f"Unknown method: {method}")


# ============================================================
# 4. SEGMENTATION METRICS
# ============================================================

def calculate_overlap_metrics(prediction, ground_truth):

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

    return {
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
    }


# ============================================================
# 5. SURFACE DISTANCE METRICS
# ============================================================

def calculate_surface_metrics(
    prediction,
    ground_truth,
    spacing,
):
    """
    Calculate HD95 and ASSD in millimetres.

    HD95:
        95th percentile of the combined bidirectional
        surface distance distributions.

    ASSD:
        Mean of the two directed mean surface distances.

    spacing:
        Physical voxel spacing in the array's spatial-axis order.
    """

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    if not prediction.any() or not ground_truth.any():
        return {
            "HD95_mm": np.nan,
            "ASSD_mm": np.nan,
        }

    structure = ndimage.generate_binary_structure(3, 1)

    pred_surface = prediction & ~ndimage.binary_erosion(
        prediction,
        structure=structure,
        border_value=0,
    )

    gt_surface = ground_truth & ~ndimage.binary_erosion(
        ground_truth,
        structure=structure,
        border_value=0,
    )

    # Distance from each ground-truth surface voxel
    # to the nearest predicted surface voxel.
    distance_to_pred = ndimage.distance_transform_edt(
        ~pred_surface,
        sampling=spacing,
    )

    # Distance from each predicted surface voxel
    # to the nearest ground-truth surface voxel.
    distance_to_gt = ndimage.distance_transform_edt(
        ~gt_surface,
        sampling=spacing,
    )

    gt_to_pred = distance_to_pred[gt_surface]
    pred_to_gt = distance_to_gt[pred_surface]

    symmetric_distances = np.concatenate([
        gt_to_pred,
        pred_to_gt,
    ])

    hd95 = np.percentile(symmetric_distances, 95)

    assd = (
        np.mean(gt_to_pred) + np.mean(pred_to_gt)
    ) / 2.0

    return {
        "HD95_mm": hd95,
        "ASSD_mm": assd,
    }


# ============================================================
# 6. LOAD A MODEL CHECKPOINT
# ============================================================

def load_model(model_name):

    checkpoint_path = CHECKPOINTS[model_name]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = MODEL_BUILDERS[model_name]().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\nLoaded {model_name}")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Best training validation Dice:",
        checkpoint["best_validation_dice"],
    )

    return model


# ============================================================
# 7. EVALUATE ONE MODEL
# ============================================================

all_results = []

for model_name in CHECKPOINTS:

    print("\n" + "=" * 65)
    print("EVALUATING:", model_name)
    print("=" * 65)

    model = load_model(model_name)

    for patient_id in tqdm(
        validation_patients,
        desc=f"{model_name} validation",
    ):

        file_path = PROCESSED_DATA_DIR / f"{patient_id}.pt"

        data = torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )

        image_meta = data["image"]

        # Obtain physical voxel spacing from the affine.
        affine = image_meta.affine.detach().cpu().numpy()

        spacing = tuple(
            np.linalg.norm(affine[:3, :3], axis=0).tolist()
        )

        image = (
            image_meta
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

        with torch.no_grad():

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
            )[0].cpu().numpy() == 1

        ground_truth = (
            label[0, 0].cpu().numpy() == 1
        )

        for method in METHODS:

            processed_prediction = apply_postprocessing(
                prediction,
                method,
            )

            overlap_metrics = calculate_overlap_metrics(
                processed_prediction,
                ground_truth,
            )

            surface_metrics = calculate_surface_metrics(
                processed_prediction,
                ground_truth,
                spacing,
            )

            all_results.append({
                "Model": model_name,
                "Patient_ID": patient_id,
                "Method": method,
                "Spacing_X_mm": spacing[0],
                "Spacing_Y_mm": spacing[1],
                "Spacing_Z_mm": spacing[2],
                **overlap_metrics,
                **surface_metrics,
            })

        del data, image, label, logits
        del prediction, ground_truth
        gc.collect()

    del model
    gc.collect()


# ============================================================
# 8. SAVE PATIENT-LEVEL RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    DETAILED_OUTPUT,
    index=False,
)


# ============================================================
# 9. SUMMARY RESULTS
# ============================================================

metric_columns = [
    "Dice",
    "IoU",
    "Precision",
    "Recall",
    "Specificity",
    "HD95_mm",
    "ASSD_mm",
]

summary_df = (
    results_df
    .groupby(["Model", "Method"])[metric_columns]
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
# 10. DISPLAY RESULTS
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
print("Patient-level results:", DETAILED_OUTPUT)
print("Summary results:", SUMMARY_OUTPUT)
