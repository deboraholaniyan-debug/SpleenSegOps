
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from dataset import SpleenPatchDataset
from model import build_model
from config import (
    PROCESSED_DATA_DIR,
    SPLIT_MANIFEST,
    MODELS_DIR,
    PATCH_SIZE,
    DEVICE,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Training configuration
EPOCHS = 20
LEARNING_RATE = 0.0005
TRAIN_SAMPLES = 528
PATIENCE = 5

device = torch.device(DEVICE)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training dataset
train_dataset = SpleenPatchDataset(
    split="train",
    samples_per_epoch=TRAIN_SAMPLES,
    foreground_probability=0.50,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

# Full-volume validation cases
manifest = pd.read_csv(SPLIT_MANIFEST)
validation_cases = manifest[
    manifest["split"] == "validation"
]

# Model
model = build_model().to(device)

# Weighted Dice + Cross-Entropy loss.
# Higher background weight penalises false-positive foreground predictions.
class_weights = torch.tensor(
    [1.5, 1.0],
    dtype=torch.float32,
    device=device,
)

criterion = DiceCELoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
    weight=class_weights,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)

best_dice = -1.0
epochs_without_improvement = 0

print("Starting revised baseline 3D U-Net training")
print("Device:", device)
print("Training patches per epoch:", len(train_dataset))
print("Full-volume validation patients:", len(validation_cases))


def calculate_dice(prediction, target):
    intersection = (
        (prediction == 1) & (target == 1)
    ).sum().float()

    denominator = (
        (prediction == 1).sum()
        + (target == 1).sum()
    ).float()

    return (
        (2.0 * intersection + 1e-6)
        / (denominator + 1e-6)
    ).item()


for epoch in range(EPOCHS):

    # -----------------------------
    # Training
    # -----------------------------
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].long().to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -----------------------------
    # Full-volume validation
    # -----------------------------
    model.eval()
    patient_dice = []

    with torch.no_grad():

        for _, row in validation_cases.iterrows():

            patient_id = row["patient_id"]

            data = torch.load(
                PROCESSED_DATA_DIR / f"{patient_id}.pt",
                map_location="cpu",
                weights_only=False,
            )

            image = data["image"].unsqueeze(0).to(device)
            target = data["label"][0].to(device)

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
            )[0]

            dice = calculate_dice(
                prediction,
                target,
            )

            patient_dice.append(dice)

    mean_dice = float(np.mean(patient_dice))

    scheduler.step(mean_dice)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Full-volume Dice: {mean_dice:.4f} "
        f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # -----------------------------
    # Save best model
    # -----------------------------
    if mean_dice > best_dice:

        best_dice = mean_dice
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_dice": best_dice,
                "seed": SEED,
                "patch_size": PATCH_SIZE,
            },
            MODELS_DIR / "baseline_3d_unet_best.pth",
        )

        print("Best checkpoint updated.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break


print("\nTraining completed.")
print(f"Best full-volume validation Dice: {best_dice:.4f}")
