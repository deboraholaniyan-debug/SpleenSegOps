from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

image_path = PROJECT_ROOT / "data/raw/imagesTr/spleen_10.nii.gz"
label_path = PROJECT_ROOT / "data/raw/labelsTr/spleen_10.nii.gz"

image = nib.load(str(image_path)).get_fdata()
label = nib.load(str(label_path)).get_fdata()

# Select the axial slice containing the largest spleen annotation
slice_counts = np.sum(label == 1, axis=(0, 1))
slice_index = int(np.argmax(slice_counts))

ct_slice = image[:, :, slice_index]
mask_slice = label[:, :, slice_index]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(ct_slice.T, cmap="gray", origin="lower")
axes[0].set_title(f"CT Scan — Slice {slice_index}")

axes[1].imshow(ct_slice.T, cmap="gray", origin="lower")
axes[1].imshow(
    np.ma.masked_where(mask_slice.T == 0, mask_slice.T),
    cmap="autumn",
    alpha=0.5,
    origin="lower"
)
axes[1].set_title("CT Scan with Spleen Annotation")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_dir = PROJECT_ROOT / "reports/figures"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "sample_spleen_visualisation.png", dpi=200)
plt.show()