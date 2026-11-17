
import torch
from dataset import SpleenPatchDataset

dataset = SpleenPatchDataset(
    split="train",
    samples_per_epoch=10,
)

sample = dataset[0]

print("Patient:", sample["patient_id"])
print("Image shape:", sample["image"].shape)
print("Label shape:", sample["label"].shape)
print("Image range:", sample["image"].min().item(),
      sample["image"].max().item())
print("Mask labels:", torch.unique(sample["label"]))
print("Spleen voxels in patch:", sample["label"].sum().item())
