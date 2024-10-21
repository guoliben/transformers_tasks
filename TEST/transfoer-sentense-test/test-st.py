from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# sentences = ["This is an example sentence", "Each sentence is converted"]
# embeddings = model.encode(sentences)
# print(embeddings)



from sklearn.metrics.pairwise import cosine_similarity
sentence1 = "This is a sentence"
sentence2 = "This is another sentence"
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
similarity = cosine_similarity([embedding1], [embedding2])
print(f"Cosine similarity: {similarity[0][0]}")