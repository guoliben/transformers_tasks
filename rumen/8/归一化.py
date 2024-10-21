import numpy as np
import matplotlib.pyplot as plt

# 生成一组数据
data = np.linspace(0, 10, 100)
data = np.linspace(0, 20, 10)

# 计算归一化后的值
min_value = np.min(data)
max_value = np.max(data)
normalized_data = (data - min_value) / (max_value - min_value)

# 绘制曲线
plt.plot(data, normalized_data)
plt.xlabel('原始数据')
plt.ylabel('归一化后的值')
plt.title('归一化曲线')
plt.show()
