import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch

model = models.mobilenet_v2(pretrained=True).features
model.eval()

image = Image.open("example.webp")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
input_tensor = preprocess(image).unsqueeze(0)
with torch.no_grad():
    features = model(input_tensor)
    vector = features.squeeze().flatten()