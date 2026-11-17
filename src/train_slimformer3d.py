
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from config import (
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
    SPLIT_MANIFEST,
    MODELS_DIR,
    PATCH_SIZE,
    DEVICE,
    RANDOM_SEED,
)

from dataset import SpleenPatchDataset
from slimformer3d_model import build_model


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

SEED = RANDOM_SEED

EPOCHS = 20
TRAIN_SAMPLES = 528
BATCH_SIZE = 1
LEARNING_RATE = 0.0005

FOREGROUND_PROBABILITY = 0.50

NUM_WORKERS = 0
PATIENCE = 6

CHECKPOINT_PATH = MODELS_DIR / "slimformer3d_best.pth"

METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "slimformer3d_validation_metrics.csv"
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY AND DEVICE
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)

torch.set_num_threads(6)

device = torch.device(DEVICE)

print("Device:", device)
print("Training samples per epoch:", TRAIN_SAMPLES)
print("Patch size:", PATCH_SIZE)


# ============================================================
# 3. DATASETS AND DATALOADERS
# ============================================================

train_dataset = SpleenPatchDataset(
    split="train",
    samples_per_epoch=TRAIN_SAMPLES,
    foreground_probability=FOREGROUND_PROBABILITY,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# Load validation patient list
split_df = pd.read_csv(SPLIT_MANIFEST)

validation_patients = split_df.loc[
    split_df["split"] == "validation",
    "patient_id"
].tolist()

print("Training patients:", len(
    split_df[split_df["split"] == "train"]
))

print("Validation patients:", len(validation_patients))


# ============================================================
# 4. MODEL, LOSS AND OPTIMIZER
# ============================================================

model = build_model().to(device)

criterion = DiceCELoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
    weight=torch.tensor([1.5, 1.0], device=device),
)

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


# ============================================================
# 5. VALIDATION FUNCTION
# ============================================================

def validate_model():
    model.eval()

    patient_dice_scores = []

    with torch.no_grad():

        for patient_id in tqdm(
            validation_patients,
            desc="Full-volume validation",
        ):

            file_path = (
                PROCESSED_DATA_DIR
                / f"{patient_id}.pt"
            )

            data = torch.load(
                file_path,
                map_location="cpu",
                weights_only=False,
            )

            image = data["image"].float().unsqueeze(0).to(device)
            label = data["label"].long().unsqueeze(0).to(device)

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
            )

            prediction = prediction[0] == 1
            ground_truth = label[0, 0] == 1

            intersection = torch.logical_and(
                prediction,
                ground_truth,
            ).sum().float()

            denominator = (
                prediction.sum().float()
                + ground_truth.sum().float()
            )

            dice = (
                (2.0 * intersection + 1e-6)
                / (denominator + 1e-6)
            )

            patient_dice_scores.append(dice.item())

            del image, label, logits, prediction, ground_truth

    mean_dice = float(np.mean(patient_dice_scores))

    return mean_dice, patient_dice_scores


# ============================================================
# 6. TRAINING LOOP
# ============================================================

best_dice = 0.0
epochs_without_improvement = 0

history = []

print("\nStarting SlimFormer3D training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
    )

    for batch in progress_bar:

        if isinstance(batch, dict):
            images = batch["image"]
            labels = batch["label"]
        else:
            images, labels = batch

        images = images.float().to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

        del images, labels, outputs, loss

    average_loss = running_loss / max(len(train_loader), 1)

    # Full-volume validation
    validation_dice, patient_scores = validate_model()

    scheduler.step(validation_dice)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f" | Loss: {average_loss:.4f}"
        f" | Validation Dice: {validation_dice:.4f}"
        f" | LR: {current_lr:.6f}"
    )

    history.append({
        "epoch": epoch + 1,
        "training_loss": average_loss,
        "validation_dice": validation_dice,
        "learning_rate": current_lr,
    })

    # Save best checkpoint
    if validation_dice > best_dice:

        best_dice = validation_dice

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_dice": best_dice,
                "config": {
                    "patch_size": PATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "train_samples": TRAIN_SAMPLES,
                    "seed": SEED,
                },
            },
            CHECKPOINT_PATH,
        )

        print(
            f"Best checkpoint saved: "
            f"{CHECKPOINT_PATH}"
        )

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break

    if device.type == "cpu":
        import gc
        gc.collect()


# ============================================================
# 7. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = (
    PROJECT_ROOT
    / "reports"
    / "slimformer3d_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False,
)

print("\nSlimFormer3D training completed.")
print("Best validation Dice:", round(best_dice, 4))
print("Checkpoint:", CHECKPOINT_PATH)
print("Training history:", history_path)
