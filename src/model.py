import torch.nn as nn
from monai.networks.nets import UNet


def build_model():
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=1,
        norm="INSTANCE",
    )


if __name__ == "__main__":
    model = build_model()
    print(model)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )