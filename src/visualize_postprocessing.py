
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from monai.inferers import sliding_window_inference

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PATCH_SIZE,
    DEVICE,
)
from model import build_model


# ==========================================
# 1. LOAD MODEL
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


# ==========================================
# 2. SELECT VALIDATION PATIENTS
# ==========================================

patients = [
    "spleen_61.nii",
    "spleen_8.nii",
]


# ==========================================
# 3. VISUALISE EACH PATIENT
# ==========================================

for patient_id in patients:

    data = torch.load(
        PROCESSED_DATA_DIR / f"{patient_id}.pt",
        map_location="cpu",
        weights_only=False,
    )

    image = data["image"].unsqueeze(0).to(device)
    label = data["label"].squeeze(0).numpy() > 0

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

    # Largest connected component
    components, number = ndimage.label(prediction)

    if number > 0:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        largest_prediction = components == np.argmax(sizes)
    else:
        largest_prediction = prediction

    # Select axial slice with largest GT spleen area
    slice_areas = label.sum(axis=(0, 1))
    slice_index = int(np.argmax(slice_areas))

    ct_slice = image[0, 0, :, :, slice_index].cpu().numpy()
    gt_slice = label[:, :, slice_index]
    original_slice = prediction[:, :, slice_index]
    filtered_slice = largest_prediction[:, :, slice_index]

    # Four-panel comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    axes[0].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("Original CT")

    axes[1].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[1].imshow(
        np.ma.masked_where(gt_slice.T == 0, gt_slice.T),
        cmap="Greens",
        alpha=0.6,
        origin="lower",
    )
    axes[1].set_title("Ground Truth")

    axes[2].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[2].imshow(
        np.ma.masked_where(original_slice.T == 0, original_slice.T),
        cmap="Reds",
        alpha=0.6,
        origin="lower",
    )
    axes[2].set_title("Original Prediction")

    axes[3].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[3].imshow(
        np.ma.masked_where(filtered_slice.T == 0, filtered_slice.T),
        cmap="Blues",
        alpha=0.6,
        origin="lower",
    )
    axes[3].set_title("Largest Component")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"Post-processing Comparison: {patient_id}",
        fontsize=14,
    )

    plt.tight_layout()

    output_path = (
        PROCESSED_DATA_DIR.parent.parent
        / "reports"
        / "figures"
        / f"{patient_id.replace('.nii', '')}_postprocessing.png"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    print("Saved:", output_path)
