import numpy as np  
  
# 假设的嵌入维度和输出维度  
embedding_dim = 4  
output_dim = 4  
  
# 随机初始化嵌入矩阵（这里仅为示例，实际中会使用预训练的嵌入或随机初始化后训练）  
embeddings = np.random.rand(2, embedding_dim)  # 假设有2个单词，“hello”和“world”  
print("Embeddings:")  
print(embeddings)  
  
# 随机初始化权重矩阵WQ, WK, WV（这里也仅为示例）  
WQ = np.random.rand(embedding_dim, output_dim)  
WK = np.random.rand(embedding_dim, output_dim)  
WV = np.random.rand(embedding_dim, output_dim)  


# 注意：这里打印的WQ, WK, WV是随机初始化的矩阵，不是训练后的值  
print("\nWeight Matrices:")  
print("WQ:")  
print(WQ)  
print("WK:")  
print(WK)  
print("WV:")  
print(WV)
  
# 生成Q, K, V向量  
Q = np.dot(embeddings, WQ)  
K = np.dot(embeddings, WK)  
V = np.dot(embeddings, WV)  
  
# 打印Q, K, V向量  
print("\nQuery Vectors (Q):")  
print(Q)  
print("\nKey Vectors (K):")  
print(K)  
print("\nValue Vectors (V):")  
print(V)  
  
