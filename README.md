# SpleenSegOps

**A CPU-based 3D medical image segmentation pipeline for spleen segmentation using deep learning and Transformer architectures.**

SpleenSegOps is a research-oriented project for developing, training, evaluating, and deploying 3D medical image segmentation models. It uses CT volumes from the Medical Segmentation Decathlon (MSD) Task09 Spleen dataset to investigate conventional convolutional and Transformer-based architectures for automatic spleen segmentation.

The project prioritises reproducibility, resource-efficient experimentation, quantitative evaluation, and deployment readiness on CPU-based environments.

> **Research prototype:** SpleenSegOps is intended for research and engineering experimentation. It has not been clinically validated and must not be used to make clinical decisions.

## Project objectives

- Develop a reproducible preprocessing pipeline for 3D CT volumes and segmentation masks.
- Train and compare 3D U-Net and Transformer-based segmentation architectures.
- Evaluate segmentation quality using overlap, classification, and surface-distance metrics.
- Investigate connected-component post-processing to improve segmentation predictions.
- Provide a foundation for an API-based inference service and containerised deployment.

## Dataset

The project uses **Medical Segmentation Decathlon – Task09 Spleen**.

The local experimental split is:

| Subset | Number of cases | Purpose |
|---|---:|---|
| Training | 33 | Model training |
| Validation | 8 | Model selection and evaluation |
| Official test | 20 | Reserved for future inference or evaluation where reference labels are available |

The official test images are unlabelled in the downloaded dataset and are excluded from the reported validation metrics.

Dataset files are not included in this repository. Obtain the dataset separately and place it in the appropriate local data directory.

## Preprocessing

The preprocessing pipeline performs the following operations:

1. Loads CT volumes and their corresponding segmentation masks.
2. Converts image orientation to RAS.
3. Resamples volumes to a target spacing of **1.0 × 1.0 × 2.5 mm**.
4. Uses bilinear interpolation for CT images and nearest-neighbour interpolation for segmentation masks.
5. Clips CT intensities to the range **−125 to 275 HU**.
6. Scales image intensities to the range [0, 1].
7. Saves processed image and mask tensors for model training and evaluation.

Training uses patch-based sampling with a patch size of **64 × 64 × 32**, batch size of 1, and CPU execution.

## Models

Three architectures have been implemented and evaluated:

| Model | Description |
|---|---|
| 3D U-Net | Convolutional encoder–decoder baseline |
| SegFormer3D | Hierarchical Transformer-based segmentation architecture adapted for the project's 3D input configuration |
| SlimFormer3D | Hybrid Transformer–CNN architecture using ClassAttention for token mixing |

The SegFormer3D and SlimFormer3D implementations are based on external research repositories and have been adapted for this experimental pipeline. Refer to their respective source directories and licence files for implementation details and attribution.

## Validation results

The following results were obtained on the eight-patient validation cohort using largest-connected-component (LCC) post-processing.

| Model | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) |
|---|---:|---:|---:|---:|---:|---:|
| 3D U-Net | 0.8908 | 0.8055 | 0.9603 | 0.8326 | 6.51 | 1.90 |
| SegFormer3D | 0.5689 | 0.4142 | 0.9659 | 0.4236 | 19.89 | 5.77 |
| SlimFormer3D | **0.9188** | **0.8515** | 0.9642 | **0.8789** | **4.16** | **1.35** |

SlimFormer3D achieved the strongest mean validation performance across the reported overlap and surface-distance metrics in this experiment.

The results should be interpreted cautiously: the validation cohort is small, and the same cohort was used during model selection and post-processing assessment. Independent evaluation is required before making broader claims about generalisation.

### Evaluation metrics

The evaluation pipeline reports:

- **Dice coefficient:** overlap between predicted and reference segmentation.
- **Intersection over Union (IoU):** intersection relative to the union of predicted and reference regions.
- **Precision:** proportion of predicted foreground voxels that are correct.
- **Recall (Sensitivity):** proportion of reference foreground voxels recovered.
- **HD95:** 95th-percentile Hausdorff surface distance, in millimetres.
- **ASSD:** average symmetric surface distance, in millimetres.

Surface-distance calculations use voxel spacing derived from the image affine.

## Repository structure

```text
SpleenSegOps/
├── data/
│   ├── raw/                  # Local dataset; excluded from Git
│   └── processed/            # Preprocessed tensors; excluded from Git
├── external/
│   ├── SegFormer3D/          # Adapted external implementation
│   └── SlimFormer3D/         # Adapted external implementation
├── notebooks/                # Experimental notebooks