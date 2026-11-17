
import random
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import PROCESSED_DATA_DIR, SPLIT_MANIFEST, PATCH_SIZE


class SpleenPatchDataset(Dataset):
    def __init__(
        self,
        split="train",
        samples_per_epoch=132,
        foreground_probability=0.67,
        cache_size=1,
    ):
        self.manifest = pd.read_csv(SPLIT_MANIFEST)
        self.manifest = self.manifest[
            self.manifest["split"] == split
        ].reset_index(drop=True)

        if self.manifest.empty:
            raise ValueError(f"No cases found for split: {split}")

        self.samples_per_epoch = samples_per_epoch
        self.foreground_probability = foreground_probability
        self.patch_size = PATCH_SIZE
        self.cache_size = cache_size

        self.cache = OrderedDict()

    def __len__(self):
        return self.samples_per_epoch

    def _load_case(self, patient_id):
        if patient_id in self.cache:
            self.cache.move_to_end(patient_id)
            return self.cache[patient_id]

        path = PROCESSED_DATA_DIR / f"{patient_id}.pt"

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        image = data["image"].float()
        label = data["label"].float()

        self.cache[patient_id] = (image, label)

        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return image, label

    def _extract_patch(self, image, label):
        _, height, width, depth = image.shape
        patch_h, patch_w, patch_d = self.patch_size

        foreground = torch.nonzero(label[0] > 0)

        use_foreground = (
            len(foreground) > 0
            and random.random() < self.foreground_probability
        )

        if use_foreground:
            selected = foreground[
                random.randrange(len(foreground))
            ]
            center_h, center_w, center_d = selected.tolist()
        else:
            center_h = random.randrange(height)
            center_w = random.randrange(width)
            center_d = random.randrange(depth)

        start_h = max(
            0, min(center_h - patch_h // 2, height - patch_h)
        )
        start_w = max(
            0, min(center_w - patch_w // 2, width - patch_w)
        )
        start_d = max(
            0, min(center_d - patch_d // 2, depth - patch_d)
        )

        image_patch = image[
            :,
            start_h:start_h + patch_h,
            start_w:start_w + patch_w,
            start_d:start_d + patch_d,
        ]

        label_patch = label[
            :,
            start_h:start_h + patch_h,
            start_w:start_w + patch_w,
            start_d:start_d + patch_d,
        ]

        padding = (
            0, max(0, patch_d - label_patch.shape[3]),
            0, max(0, patch_w - label_patch.shape[2]),
            0, max(0, patch_h - label_patch.shape[1]),
        )

        if any(padding):
            image_patch = F.pad(image_patch, padding)
            label_patch = F.pad(label_patch, padding)

        return image_patch.contiguous(), label_patch.contiguous()

    def __getitem__(self, index):
        row = self.manifest.sample(n=1).iloc[0]
        patient_id = row["patient_id"]

        image, label = self._load_case(patient_id)
        image_patch, label_patch = self._extract_patch(
            image, label
        )

        return {
            "image": image_patch,
            "label": label_patch,
            "patient_id": patient_id,
        }
