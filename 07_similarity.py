from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love cats.",
    "I really like kittens",
    "The stock market crashed"
]

embeddings = model.encode(sentences)

similarity = cosine_similarity(embeddings)

print(similarity)