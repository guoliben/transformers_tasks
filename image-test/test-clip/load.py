from chinese_clip import load_model

model, preprocess = load_model('ViT-B-16', device='cpu')  # or 'cuda'
print("模型加载成功")


# pip install git+https://github.com/OFA-Sys/Chinese-CLIP.git

# python -c "from chinese_clip import load_model; print('✅ 模块可用')"

# python -c "from Chinese_CLIP import load_model; print('✅ 模块可用')"


# pip install cn_clip