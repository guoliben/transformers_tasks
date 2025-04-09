import torch
from torchvision import models, transforms
from PIL import Image

# 加载预训练的 MobileNet 模型
model = models.mobilenet_v2(pretrained=True)
model.eval()  # 设置为推理模式

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# 加载并处理图像
img = Image.open("example.webp").convert("RGB")
input_tensor = transform(img).unsqueeze(0)  # 增加 batch 维度

# 推理
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

# 获取类别名称
from torchvision.models import mobilenet_v2
from torchvision.datasets.utils import download_url
import json

# 下载ImageNet标签
# url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
# download_url(url, '.', 'imagenet_classes.txt')

with open("imagenet_classes2.txt") as f:
    # labels = [line.strip() for line in f.readlines()]
    # print(labels)
    labels = []
    for line in f.readlines():
        cleaned_line = line.strip()
        # print("读取类别:", cleaned_line)
        labels.append(cleaned_line)

print("识别结果：", labels[predicted.item()])

print(predicted.item())