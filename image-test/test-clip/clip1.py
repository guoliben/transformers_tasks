# from chinese_clip import load_model, tokenize

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models, tokenize

print("Available models:", available_models())


from PIL import Image
import torch
from torchvision import transforms

# 1. 加载模型
model, preprocess = load_from_name('ViT-B-16', device='cpu')  # 可改为 'cuda'

# 2. 加载图片
image = Image.open("dog1.jpg").convert("RGB")
image = Image.open("dog2.jpg").convert("RGB")
image = Image.open("cat1.png").convert("RGB")

image_input = preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)

# 3. 输入文本
texts = ["猫", "狗", "人", "建筑"]  # 支持多文本匹配
text_tokens = tokenize(texts)

# 4. 提取特征并计算相似度
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = image_features @ text_features.T

# 5. 输出最相关文本
best_match = similarity[0].argmax()
print(f"预测内容：{texts[best_match]}，相似度：{similarity[0][best_match].item():.4f}")