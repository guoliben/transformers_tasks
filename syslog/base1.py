import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import classification_report  
  
# 假设我们有一个包含日志消息和标签的DataFrame  
data = pd.DataFrame({  
    'log_message': ['Error in connection', 'User login successful', 'Memory usage high'],  
    'label': ['error', 'success', 'warning']  
})  
  
# 分割数据  
X_train, X_test, y_train, y_test = train_test_split(data['log_message'], data['label'], test_size=0.2, random_state=42)  
 
print("----------1")
print(X_train, X_test, y_train, y_test) 
# 特征提取  
vectorizer = TfidfVectorizer()  
X_train_tfidf = vectorizer.fit_transform(X_train)  
X_test_tfidf = vectorizer.transform(X_test)  
  
print("----------2")
print(vectorizer)
print(X_train_tfidf,X_test_tfidf )


# 训练模型  
model = MultinomialNB()  
model.fit(X_train_tfidf, y_train)  

print("----------3")
print(model)

# 预测与评估  
print("----------4")
y_pred = model.predict(X_test_tfidf)  
print(classification_report(y_test, y_pred))


