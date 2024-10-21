import numpy as np

def calculate_cosine_similarity(doc1, doc2):
    # 计算两者之间的余弦相似性
    dot_product = np.dot(doc1, doc2)
    magnitude_doc1 = np.linalg.norm(doc1)
    magnitude_doc2 = np.linalg.norm(doc2)

    similarity = dot_product / (magnitude_doc1 * magnitude_doc2)

    return similarity

# 假设我们有两个文档的向量表示
doc1_vector = [0.5, 0.3, -0.4]
doc2_vector = [-0.7, 0.8, 0.1]

similarity = calculate_cosine_similarity(doc1_vector, doc2_vector)

print(f"文档1和文档2的相似度为：{similarity}")