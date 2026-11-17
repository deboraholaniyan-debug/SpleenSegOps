
import torch
import numpy as np
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference

from config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PATCH_SIZE,
    DEVICE,
)
from model import build_model

patient_id = "spleen_8.nii"

device = torch.device(DEVICE)

checkpoint = torch.load(
    MODELS_DIR / "baseline_3d_unet_best.pth",
    map_location=device,
    weights_only=False,
)

model = build_model().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

data = torch.load(
    PROCESSED_DATA_DIR / f"{patient_id}.pt",
    map_location="cpu",
    weights_only=False,
)

image = data["image"].unsqueeze(0).to(device)
label = data["label"].squeeze(0).numpy()

with torch.no_grad():
    logits = sliding_window_inference(
        image,
        roi_size=PATCH_SIZE,
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
    )

    prediction = torch.argmax(
        logits, dim=1
    )[0].cpu().numpy()

# Select the axial slice with the largest ground-truth spleen area
slice_areas = label.sum(axis=(0, 1))
slice_index = int(np.argmax(slice_areas))

ct_slice = image[0, 0, :, :, slice_index].cpu().numpy()
gt_slice = label[:, :, slice_index]
pred_slice = prediction[:, :, slice_index]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(ct_slice.T, cmap="gray", origin="lower")
axes[0].set_title("Original CT")

axes[1].imshow(ct_slice.T, cmap="gray", origin="lower")
axes[1].imshow(
    np.ma.masked_where(gt_slice.T == 0, gt_slice.T),
    cmap="Greens",
    alpha=0.6,
    origin="lower",
)
axes[1].set_title("Ground-truth spleen")

axes[2].imshow(ct_slice.T, cmap="gray", origin="lower")
axes[2].imshow(
    np.ma.masked_where(pred_slice.T == 0, pred_slice.T),
    cmap="Reds",
    alpha=0.6,
    origin="lower",
)
axes[2].set_title("Predicted spleen")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "reports/figures/baseline_prediction.png",
    dpi=200,
)
plt.show()
