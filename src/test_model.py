import torch
from model import build_model

model = build_model()
model.eval()

x = torch.randn(1, 1, 64, 64, 32)

with torch.no_grad():
    output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output finite:", torch.isfinite(output).all().item())