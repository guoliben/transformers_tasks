import numpy as np
from math import cos

# 定义两个向量
vector1 = [210, 230,1321, 230,1321]
vector2 = [992,4,11]
vector2 = [200, 230,1321]

# 计算两个向量的点积（内积）
dot_product = np.dot(vector1, vector2)

# 计算两个向量的长度（模长）
length_vector1 = np.linalg.norm(vector1)
length_vector2 = np.linalg.norm(vector2)

# 计算余弦相似度
cosine_similarity = dot_product / (length_vector1 * length_vector2)

print("余弦相似度: ", cosine_similarity)