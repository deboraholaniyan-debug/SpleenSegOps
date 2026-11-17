# SpleenSegOps

**CPU-Based 3D Medical Image Segmentation Using Convolutional and Transformer Architectures**

SpleenSegOps is a research-oriented pipeline for automatic spleen segmentation from 3D CT volumes. It uses the Medical Segmentation Decathlon (MSD) Task09 Spleen dataset to develop, train, compare, evaluate and deploy deep-learning segmentation models.

The project investigates conventional convolutional networks and Transformer-based architectures, with an emphasis on reproducible experimentation, resource-efficient model execution, quantitative evaluation and containerised inference.

> **Research prototype:** SpleenSegOps has not undergone clinical validation. It is intended for research and engineering experimentation and must not be used for diagnosis or clinical decision-making.

## 1. Project objectives

- Develop a reproducible preprocessing pipeline for 3D CT volumes and segmentation masks.
- Implement and compare 3D U-Net, SegFormer3D and SlimFormer3D.
- Evaluate segmentation performance using overlap, classification and surface-distance metrics.
- Investigate connected-component post-processing.
- Track model experiments and evaluation results using MLflow.
- Provide a CPU-based inference service using FastAPI and Docker.

## 2. Dataset

The project uses the **Medical Segmentation Decathlon – Task09 Spleen** dataset.

The experimental split consists of:

| Subset | Cases | Purpose |
|---|---:|---|
| Training | 33 | Model training |
| Validation | 8 | Model selection and evaluation |
| Official test | 20 | Reserved for future inference or evaluation when reference labels are available |

The official test images are unlabelled in the downloaded dataset and were excluded from the reported validation results.

Dataset files are not distributed with this repository. Obtain the dataset separately and place it in `data/raw/`.

## 3. Preprocessing

The preprocessing pipeline uses MONAI and PyTorch to perform:

1. CT volume and segmentation-mask loading.
2. Orientation standardisation to RAS.
3. Resampling to 1.0 × 1.0 × 2.5 mm voxel spacing.
4. Bilinear interpolation for CT images and nearest-neighbour interpolation for segmentation masks.
5. CT intensity clipping to −125 to 275 HU.
6. Intensity scaling to [0, 1].
7. Saving of preprocessed image and mask tensors.

Training uses patch-based sampling with a patch size of 64 × 64 × 32 and a batch size of 1.

## 4. Model architectures

Three segmentation architectures were evaluated.

| Model | Architecture |
|---|---|
| 3D U-Net | Convolutional encoder–decoder baseline |
| SegFormer3D | Hierarchical Transformer-based segmentation architecture adapted for the project's 3D input configuration |
| SlimFormer3D | Hybrid Transformer–CNN architecture using ClassAttention for token mixing |

The SegFormer3D and SlimFormer3D implementations were adapted from their respective external research repositories. Their original sources and licences should be consulted for implementation attribution and reuse conditions.

## 5. Experimental results

Evaluation was conducted on the eight-patient validation cohort. The following table reports the results after largest-connected-component (LCC) post-processing.

| Model | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) |
|---|---:|---:|---:|---:|---:|---:|
| 3D U-Net | 0.8908 | 0.8055 | 0.9603 | 0.8326 | 6.51 | 1.90 |
| SegFormer3D | 0.5689 | 0.4142 | 0.9659 | 0.4236 | 19.89 | 5.77 |
| **SlimFormer3D** | **0.9188** | **0.8515** | 0.9642 | **0.8789** | **4.16** | **1.35** |

SlimFormer3D achieved the strongest mean validation performance across the reported overlap and surface-distance metrics in this experiment.

The results are preliminary. The validation cohort is small, and it was used for both model selection and post-processing assessment. Independent evaluation is necessary before making broader generalisation claims.

### Evaluation metrics

- **Dice coefficient:** Measures overlap between the predicted and reference segmentation.
- **Intersection over Union (IoU):** Measures the intersection relative to the union of predicted and reference regions.
- **Precision:** Measures the proportion of predicted foreground voxels that are correct.
- **Recall (Sensitivity):** Measures the proportion of reference foreground voxels recovered.
- **HD95:** Measures the 95th-percentile Hausdorff surface distance in millimetres.
- **ASSD:** Measures the average symmetric surface distance in millimetres.

Surface-distance calculations use voxel spacing derived from the image affine.

## 6. Repository structure

```text
SpleenSegOps/
├── data/
│   ├── raw/                    # Local dataset; excluded from Git
│   └── processed/              # Preprocessed tensors; excluded from Git
├── external/
│   ├── SegFormer3D/
│   └── SlimFormer3D/
├── notebooks/
├── src/
│   ├── config.py
│   ├── create_data_split.py
│   ├── preprocess.py
│   ├── train.py
│   ├── train_segformer3d.py
│   ├── train_slimformer3d.py
│   ├── evaluate_all_models.py
│   ├── inference.py
│   ├── api.py
│   └── track_experiments.py
├── models/                     # Local model checkpoints; excluded from Git
├── reports/
│   ├── figures/
│   └── evaluation CSV files
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## 7. Installation and execution

### Requirements

- Python 3.11 or a compatible Python environment
- PyTorch CPU
- MONAI
- Docker Desktop (for containerised inference)
- Medical Segmentation Decathlon Task09 Spleen dataset

### Clone the repository

```bash
git clone https://github.com/deboraholaniyan-debug/SpleenSegOps.git
cd SpleenSegOps
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install PyTorch CPU and project dependencies:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Place the downloaded dataset in `data/raw/`.

### Prepare the dataset

```bash
python src/create_data_split.py
python src/preprocess.py
```

### Train the models

```bash
python src/train.py
python src/train_segformer3d.py
python src/train_slimformer3d.py
```

### Evaluate the models

```bash
python src/evaluate_all_models.py
```

The evaluation scripts generate comparative metrics and summary CSV files under `reports/`.

## 8. MLflow experiment tracking

SpleenSegOps uses MLflow with a local SQLite tracking database.

The existing training and evaluation records can be imported into MLflow without retraining:

```bash
python src/track_experiments.py
```

This creates retrospective experiment runs for the three evaluated architectures and records available training histories, validation metrics and checkpoint artifacts.

Launch the MLflow interface:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

`http://localhost:5000`

**Note:** The imported runs document completed experiments retrospectively. They do not mean that MLflow was active during the original model-training sessions.

## 9. Inference using FastAPI

The project provides a FastAPI inference service that accepts a NIfTI CT volume and returns a predicted spleen segmentation mask.

Run locally:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8002
```

API documentation:

`http://localhost:8002/docs`

Available endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API health and model information |
| `/predict` | POST | Upload a NIfTI CT volume and receive a predicted segmentation mask |

The inference pipeline uses the trained SlimFormer3D checkpoint and returns a NIfTI mask mapped back to the original CT image grid.

## 10. Docker deployment

The API can be packaged into a Docker image for CPU-based deployment.

Build the image:

```bash
docker build -t spleensegops:1.0 .
```

Run the container:

```bash
docker run -d --name spleensegops-api -p 8002:8000 spleensegops:1.0
```

Check the running container:

```bash
docker ps
```

Access the API:

`http://localhost:8002/docs`

The Docker build requires the trained checkpoint at:

`models/slimformer3d_best.pth`

Model checkpoints and medical imaging data are excluded from the Git repository. The checkpoint must therefore be made available locally before building the image.

## 11. Limitations

The following limitations should be considered when interpreting the project:

1. **Small validation cohort:** Model comparison was conducted using only eight labelled validation cases.
2. **No independent labelled test evaluation:** The downloaded official test set does not include reference segmentation labels.
3. **Limited generalisability:** Results from one dataset and organ do not establish performance on other organs, scanners or clinical populations.
4. **CPU execution:** CPU-only training and inference impose computational constraints compared with GPU-based experimentation.
5. **Post-processing dependence:** Largest-connected-component post-processing improves the reported segmentation results, but its effectiveness may vary across cases.
6. **External architecture adaptations:** The adapted Transformer implementations may differ from their original configurations.
7. **No clinical validation:** The system has not been evaluated for clinical safety or diagnostic use.

## 12. Reproducibility and responsible use

- The experimental split is generated using a fixed random seed.
- Preprocessing parameters and model configurations are defined in the source code.
- Evaluation results are stored as CSV files.
- MLflow records the available experiment histories and evaluation metrics.
- Dataset files and trained checkpoints are excluded from the public repository.

Researchers reproducing the experiments should obtain the original dataset, follow its access and usage conditions, and ensure that the required model checkpoints are available.

## 13. Acknowledgements and references

SpleenSegOps builds on the Medical Segmentation Decathlon Task09 Spleen dataset and external research implementations of SegFormer3D and SlimFormer3D.

Please consult the original dataset publication, architecture papers and source repositories when using or extending this project.

## 14. Disclaimer

SpleenSegOps is an academic research and engineering prototype. It is not a certified medical device, has not undergone clinical validation, and must not be used as a substitute for professional medical assessment.

## 15. Author

**Dr Deborah Olaniyan**  
Artificial Intelligence Researcher 

GitHub: https://github.com/deboraholaniyan-debug
