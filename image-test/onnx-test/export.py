import torch
from torchvision import models

model = models.mobilenet_v2(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "mobilenetv2.onnx", input_names=["input"], output_names=["output"])