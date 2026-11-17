
from pathlib import Path

import pandas as pd
import mlflow


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment("SpleenSegOps_Model_Comparison")

models = {
    "3D_UNet": {
        "history": None,
        "checkpoint": "baseline_3d_unet_best.pth",
    },
    "SegFormer3D": {
        "history": "segformer3d_training_history.csv",
        "checkpoint": "segformer3d_best.pth",
    },
    "SlimFormer3D": {
        "history": "slimformer3d_training_history.csv",
        "checkpoint": "slimformer3d_best.pth",
    },
}

summary = pd.read_csv(REPORTS / "all_models_surface_summary.csv")

for model_name, config in models.items():

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model", model_name)
        mlflow.log_param("device", "CPU")
        mlflow.log_param("input_channels", 1)
        mlflow.log_param("num_classes", 2)
        mlflow.log_param("patch_size", "64x64x32")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "DiceCELoss")
        mlflow.log_param("learning_rate", 0.0005)

        # Import training history where available
        history_path = config["history"]

        if history_path:
            history_file = REPORTS / history_path
            history = pd.read_csv(history_file)

            mlflow.log_artifact(str(history_file))

            mlflow.log_param("epochs_completed", len(history))

            for _, row in history.iterrows():
                step = int(row["epoch"])

                mlflow.log_metrics(
                    {
                        "training_loss": float(row["training_loss"]),
                        "validation_dice": float(row["validation_dice"]),
                        "learning_rate": float(row["learning_rate"]),
                    },
                    step=step,
                )

            best_row = history.loc[history["validation_dice"].idxmax()]

            mlflow.log_metric(
                "best_validation_dice",
                float(best_row["validation_dice"]),
            )

        # Import evaluation metrics
        model_results = summary[
            summary["Model"].str.replace(" ", "_") == model_name
        ]

        for _, row in model_results.iterrows():

            method = row["Method"]

            for metric in [
                "Dice",
                "IoU",
                "Precision",
                "Recall",
                "Specificity",
                "HD95_mm",
                "ASSD_mm",
            ]:
                mlflow.log_metric(
                    f"{method}_{metric}",
                    float(row[metric]),
                )

        # Record checkpoint as an artifact
        checkpoint = MODELS / config["checkpoint"]

        if checkpoint.exists():
            mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")

        # Preserve evaluation summary
        mlflow.log_artifact(
            str(REPORTS / "all_models_surface_summary.csv")
        )

        print(f"Logged MLflow run: {model_name}")

print("\nMLflow experiment tracking completed.")
