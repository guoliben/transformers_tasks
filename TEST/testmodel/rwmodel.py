import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# 生成一些数据  
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# 创建并训练模型  
model = LinearRegression()
model.fit(X, y)

# 保存模型到文件  
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 加载模型  
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 使用加载的模型进行预测  
X_new = np.array([[6]])
y_pred = loaded_model.predict(X_new)
print(f'Predicted value for X={6}: {y_pred[0]}')