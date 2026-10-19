
from nltk.lm import Laplace, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from collections import Counter

corpus = [
    "I love cats",
    "I love cats",
    "I love dogs",
    "You love cats",
    "You love NLP"
]

tokens = [s.split() for s in corpus]

# Bigram modelleri
def train(model):
    train_data, vocab = padded_everygram_pipeline(2, tokens)
    model.fit(train_data, vocab)
    return model

laplace = train(Laplace(2))
kneser_ney = train(KneserNeyInterpolated(2))

# Good-Turing: görülmeyen olaylara ayrılan olasılık
bigrams = []
for sentence in tokens:
    bigrams += list(zip(sentence[:-1], sentence[1:]))

counts = Counter(bigrams)
frequency_counts = Counter(counts.values())

N1 = frequency_counts[1]
N = sum(counts.values())

good_turing_unseen = N1 / N

# Karşılaştırma
print("Prediction: P(cats | love)\n")

print("Laplace:",
      round(laplace.score("cats", ["love"]), 4))

print("Kneser-Ney:",
      round(kneser_ney.score("cats", ["love"]), 4))

print("\nGood-Turing:")
print("Unseen probability mass:",
      round(good_turing_unseen, 4))
