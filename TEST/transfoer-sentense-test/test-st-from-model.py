import pickle
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#
# sentences = ["This is an example sentence", "Each sentence is converted"]
# embeddings = model.encode(sentences)
# print(embeddings)
#
# # 保存模型到文件
# with open('model-st.pkl', 'wb') as f:
#     pickle.dump(model, f)

# from sklearn.metrics.pairwise import cosine_similarity
# sentence1 = "This is a sentence"
# sentence2 = "This is another sentence"
# embedding1 = model.encode(sentence1)
# embedding2 = model.encode(sentence2)
# similarity = cosine_similarity([embedding1], [embedding2])
# print(f"Cosine similarity: {similarity[0][0]}")



# 加载模型
with open('model-st.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
#
from sklearn.metrics.pairwise import cosine_similarity
sentence1 = "abc"
sentence2 = "abc"
embedding1 = loaded_model.encode(sentence1)
embedding2 = loaded_model.encode(sentence2)
similarity = cosine_similarity([embedding1], [embedding2])
print(f"Cosine similarity: {similarity[0][0]}")

