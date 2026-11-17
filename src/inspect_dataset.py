from pathlib import Path
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "imagesTr"
LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "labelsTr"

image_files = sorted(
    p for p in IMAGE_DIR.glob("*.nii.gz")
    if not p.name.startswith("._")
)

print(f"{'Volume':<20} {'Shape':<22} {'Spacing (mm)':<35} {'Spleen voxels':>15}")
print("-" * 100)

for image_path in image_files:
    label_path = LABEL_DIR / image_path.name

    image = nib.load(str(image_path))
    label = nib.load(str(label_path))

    mask_data = np.asanyarray(label.dataobj)
    spleen_voxels = int(np.count_nonzero(mask_data == 1))

    shape = str(image.shape)
    spacing = str(tuple(round(float(x), 3) for x in image.header.get_zooms()[:3]))

    print(
        f"{image_path.name:<20} "
        f"{shape:<22} "
        f"{spacing:<35} "
        f"{spleen_voxels:>15,}"
    )