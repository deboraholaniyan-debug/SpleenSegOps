
from pathlib import Path

import pandas as pd
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLIT_MANIFEST,
    TARGET_SPACING,
    CT_INTENSITY_MIN,
    CT_INTENSITY_MAX,
)


def get_preprocessing_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(
            keys=["image", "label"],
            axcodes="RAS"
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=CT_INTENSITY_MIN,
            a_max=CT_INTENSITY_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])


def main():
    manifest = pd.read_csv(SPLIT_MANIFEST)

    # Process labelled training and validation cases only.
    labelled_cases = manifest[
        manifest["split"].isin(["train", "validation"])
    ]

    transforms = get_preprocessing_transforms()

    for _, row in labelled_cases.iterrows():
        case = {
            "image": str(RAW_DATA_DIR / row["image"]),
            "label": str(RAW_DATA_DIR / row["label"]),
        }

        processed = transforms(case)

        patient_id = row["patient_id"]

        output_path = PROCESSED_DATA_DIR / f"{patient_id}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import torch

        torch.save(
            {
                "image": processed["image"].cpu(),
                "label": processed["label"].cpu(),
                "patient_id": patient_id,
                "split": row["split"],
            },
            output_path,
        )

        print(
            f"Processed {patient_id}: "
            f"image={tuple(processed['image'].shape)}, "
            f"label={tuple(processed['label'].shape)}"
        )

    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
