from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-miniLM-L6-v2")

sentences = [
    "I love cats.",
    "I really like kittens.",
    "The stock market crashed."
]

embeddings = model.encode(sentences)

print("Shape:")
print(embeddings.shape)

print("\nFirst sentence vector:")
print(embeddings[0])