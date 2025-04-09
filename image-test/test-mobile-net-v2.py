import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 加载 MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

image = Image.open("2025-04-09_15-37-57.png").convert("RGB")
image = Image.open("boat.jpg").convert("RGB")
image = Image.open("people.jpg").convert("RGB")

input_tensor = transform(image).unsqueeze(0)
# 模型推理
with torch.no_grad():
    output = model(input_tensor)

# 获取标签
with open("imagenet_classes2.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# 打印 Top-5 预测标签
_, indices = torch.topk(output, 10)
for i in indices[0]:
    print(labels[i])