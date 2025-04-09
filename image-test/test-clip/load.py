from chinese_clip import load_model

model, preprocess = load_model('ViT-B-16', device='cpu')  # or 'cuda'
print("模型加载成功")