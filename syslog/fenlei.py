from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import numpy as np
import os
import joblib

# 设置 MPS 后端（如果适用）
device = torch.device("mps")

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# 从文件中读取文本数据（如果有新文件路径可以修改这里）
file_path = 'your_text_file.txt'
texts = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        texts.append(line.strip())

# 检查是否已有保存的聚类模型
clustering_model_path = 'clustering_model.pkl'
if os.path.exists(clustering_model_path):
    # 加载已保存的聚类模型
    kmeans = joblib.load(clustering_model_path)
else:
    # 对文本进行编码
    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # 转换为 numpy 数组以便进行聚类
    embeddings_np = embeddings.cpu().numpy()

    # 进行聚类
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(embeddings_np)

    # 保存聚类模型
    joblib.dump(kmeans, clustering_model_path)

# 使用聚类模型进行预测
labels = kmeans.predict(embeddings_np)

# 输出聚类结果
for i, text in enumerate(texts):
    print(f"Text: {text}, Cluster: {labels[i]}")

# 使用分类器对新文本进行分类
new_text = "This is a new text for classification."
encoded_input = tokenizer([new_text], padding=True, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    new_output = model(**encoded_input)
new_embedding = new_output.last_hidden_state.mean(dim=1)
new_embedding_np = new_embedding.cpu().numpy()
new_label = kmeans.predict(new_embedding_np)
print(f"New text belongs to cluster: {new_label[0]}")
