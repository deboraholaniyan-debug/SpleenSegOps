
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensityRange,
)
from monai.inferers import sliding_window_inference

from config import (
    MODELS_DIR,
    TARGET_SPACING,
    CT_INTENSITY_MIN,
    CT_INTENSITY_MAX,
    PATCH_SIZE,
    DEVICE,
)

from slimformer3d_model import build_model


CHECKPOINT_PATH = MODELS_DIR / "slimformer3d_best.pth"


def load_model():
    """Load the trained SlimFormer3D checkpoint."""

    device = torch.device(DEVICE)
    model = build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded SlimFormer3D checkpoint from epoch {checkpoint['epoch']}")

    return model, device


def preprocess_ct(input_path):
    """Apply the same preprocessing used during training."""

    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS", labels=None),
        Spacing(
            pixdim=TARGET_SPACING,
            mode="bilinear",
        ),
        ScaleIntensityRange(
            a_min=CT_INTENSITY_MIN,
            a_max=CT_INTENSITY_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ])

    return transforms(str(input_path))


def predict_spleen(input_path, output_path):
    """Generate a spleen mask aligned with the original CT scan."""

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"CT scan not found: {input_path}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Preserve original CT geometry.
    original_ct = nib.load(str(input_path))
    original_shape = original_ct.shape[:3]
    original_affine = original_ct.affine
    original_header = original_ct.header.copy()

    model, device = load_model()

    # Preprocess CT.
    image = preprocess_ct(input_path)
    preprocessed_affine = image.affine.cpu().numpy()

    image = image.unsqueeze(0).to(device)

    # Run sliding-window inference.
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=image,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

        prediction = torch.argmax(logits, dim=1)

    prediction = (
        prediction.squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # Create segmentation in preprocessed geometry.
    processed_mask = nib.Nifti1Image(
        prediction,
        preprocessed_affine,
    )

    # Resample segmentation back to original CT geometry.
    native_mask = resample_from_to(
        processed_mask,
        (original_shape, original_affine),
        order=0,
    )

    native_mask_data = (
        np.asarray(native_mask.dataobj)
        .astype(np.uint8)
    )

    # Save mask using original CT geometry.
    original_header.set_data_dtype(np.uint8)

    output_image = nib.Nifti1Image(
        native_mask_data,
        original_affine,
        header=original_header,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_image, str(output_path))

    print("Segmentation completed.")
    print("Output:", output_path)
    print("Original CT shape:", original_shape)
    print("Output mask shape:", native_mask_data.shape)
    print("Predicted spleen voxels:", int(native_mask_data.sum()))

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SlimFormer3D spleen segmentation inference"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="reports/predictions/spleen_mask.nii.gz",
    )

    args = parser.parse_args()

    predict_spleen(
        input_path=args.input,
        output_path=args.output,
    )
