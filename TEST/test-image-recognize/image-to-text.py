from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import requests
from PIL import Image

model = AutoModelForCausalLM.from_pretrained('ucsahin/TraVisionLM-Object-Detection-ft', trust_remote_code=True, device_map="mps")
# you can also load the model in bfloat16 or float16
# model = AutoModelForCausalLM.from_pretrained('ucsahin/TraVisionLM-base', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
processor = AutoProcessor.from_pretrained('ucsahin/TraVisionLM-Object-Detection-ft', trust_remote_code=True)

url = "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = Image.open("car.jpg").convert("RGB")
prompt = "İşaretle: araba"
# prompt = "Tespit et: araba"

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9, top_k=50, repetition_penalty=1.2)

output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print("Model response: ", output_text)
"""
Model response:  İşaretle: araba
<loc0048><loc0338><loc0912><loc0819> araba;
"""
