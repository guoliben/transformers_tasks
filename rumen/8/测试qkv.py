import numpy as np


# 定义 QKV 矩阵的维度
qkv_dim = 512


# 生成 Query 矩阵
query = np.random.uniform(-np.sqrt(1/qkv_dim), np.sqrt(1/qkv_dim), size=(qkv_dim, qkv_dim))


# 生成 Key 矩阵
key = np.random.uniform(-np.sqrt(1/qkv_dim), np.sqrt(1/qkv_dim), size=(qkv_dim, qkv_dim))


# 生成 Value 矩阵
value = np.random.uniform(-np.sqrt(1/qkv_dim), np.sqrt(1/qkv_dim), size=(qkv_dim, qkv_dim))


# 定义头数和头大小
head_num = 8
head_size = qkv_dim // head_num


# 转换 QKV 矩阵到 Transformer 标准模式
#qkv = np.zeros((head_num, head_size, qkv_dim))
qkv = np.zeros((head_num, qkv_dim,  head_size))
for i in range(head_num):
    qkv[i] = query[:, i*head_size:(i+1)*head_size]
    #qkv[i] = query[:, :, i*head_size:(i+1)*head_size]


# 生成权重矩阵
weight = np.random.uniform(-np.sqrt(1/head_num), np.sqrt(1/head_num), size=(head_num, head_size))


# 进行 QKV 计算
result = qkv @ weight.T


print(result)

