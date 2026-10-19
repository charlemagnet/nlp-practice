
from collections import Counter

# Training corpus
corpus = [
    "I love cats",
    "I love dogs",
    "I love NLP",
    "You love cats"
]

sentences = [sentence.split() for sentence in corpus]


# N-gram oluşturma
def get_ngrams(tokens, n):
    return [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]


# Unigram, Bigram, Trigram
for n in [1, 2, 3]:
    print(f"\n--- {n}-GRAM ---")

    all_ngrams = []

    for tokens in sentences:
        all_ngrams.extend(get_ngrams(tokens, n))

    counts = Counter(all_ngrams)

    for ngram, count in counts.items():
        print(" ".join(ngram), "->", count)


# Trigram ile sonraki kelime tahmini
print("\n--- NEXT WORD PREDICTION ---")

context = ("I", "love")

bigrams = Counter()
trigrams = Counter()

for tokens in sentences:
    bigrams.update(get_ngrams(tokens, 2))
    trigrams.update(get_ngrams(tokens, 3))

print("Input: I love ___")

for trigram, count in trigrams.items():
    if trigram[:2] == context:
        next_word = trigram[2]
        probability = count / bigrams[context]

        print(f"P({next_word} | I love) = {probability:.2%}")

print("\n--- PROBABILITY COMPARISON ---")

unigrams = Counter()

for tokens in sentences:
    unigrams.update(get_ngrams(tokens, 1))

# Unigram: P(cats)
p_unigram = unigrams[("cats",)] / sum(unigrams.values())

# Bigram: P(cats | love)
p_bigram = bigrams[("love", "cats")] / unigrams[("love",)]

# Trigram: P(cats | I love)
p_trigram = trigrams[("I", "love", "cats")] / bigrams[("I", "love")]

print(f"Unigram P(cats)          = {p_unigram:.2%}")
print(f"Bigram  P(cats | love)   = {p_bigram:.2%}")
print(f"Trigram P(cats | I love) = {p_trigram:.2%}")