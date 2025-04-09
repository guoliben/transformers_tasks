from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# 加载图片
image = Image.open("boat.jpg").convert("RGB")
image = Image.open("people.jpg").convert("RGB")
image = Image.open("window.png").convert("RGB")
image = Image.open("2025-04-09_15-37-57.png").convert("RGB")

image = Image.open("cat1.png").convert("RGB")
# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 推理
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("图片描述:", caption)