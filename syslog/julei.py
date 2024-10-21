from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import numpy as np
import os

# 设置 MPS 后端
device = torch.device("mps")

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# 从文件中读取文本数据
file_path = '/Users/ben/syslog-ana/dlp.log'
file_path = '/Users/ben/nlp/auth.log'
file_path = '/Users/ben/nlp/syslog-ana/.log'
file_path = '/Users/ben/ollama/transformers_tasks/syslog/auth-event-template.log'
file_path = '/Users/ben/nlp/auth.log'
file_path = '/Users/ben/nlp/auth.log-one-julei'
texts = []
l = 0
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        l = l +1
        if l > 1000:
            break
        texts.append(line.strip())


#texts = ['fdafddfa', 'qedfqfdkjfdf', 'fdafe32453543', '5423223', '312039']
# 对文本进行编码
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**encoded_inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)

# 转换为 numpy 数组以便进行聚类
embeddings_np = embeddings.cpu().numpy()

# 使用 K-Means 进行聚类
kmeans = KMeans(n_clusters=2).fit(embeddings_np)
labels = kmeans.labels_

# 输出聚类结果
with open("julei-auth.txt", "w") as f:
    for i, text in enumerate(texts):
        f.write(f"Text: {text}, Cluster: {labels[i]}")
        print(f"Text: {text}, Cluster: {labels[i]}")
