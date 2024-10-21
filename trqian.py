from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "hello world"
inputs = tokenizer(text, return_tensors="pt")


outputs = model(**inputs)
word_embeddings = outputs.last_hidden_state


print(word_embeddings)
