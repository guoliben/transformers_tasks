from transformers import pipeline

generator = pipeline("text-generation")
results = generator("To be a billion,  I will teach you how")
print(results)
results = generator(
    "To be a billion,  I will teach you how",
    num_return_sequences=2,
    max_length=50
)
print(results)

#中文 lan