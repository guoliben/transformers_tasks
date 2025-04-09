import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys

# ========== 动态图片路径输入 ==========
if len(sys.argv) != 2:
    print("Usage: python onnx_infer.py <image_path>")
    exit(1)

image_path = sys.argv[1]

# ========== 图像预处理 ==========
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).numpy()  # 转成 numpy 格式

# ========== 加载 ONNX 模型 ==========
session = ort.InferenceSession("mobilenetv2.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_tensor})[0]  # 模型输出

# ========== 加载标签 ==========
with open("imagenet_classes2.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# ========== Top-K 输出 ==========
topk = 5
top_indices = np.argsort(output[0])[::-1][:topk]
print("📸 图像预测 Top-{}:".format(topk))
for idx in top_indices:
    print(f"{labels[idx]}  ({output[0][idx]:.4f})")