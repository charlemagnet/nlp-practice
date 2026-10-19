from collections import defaultdict, Counter

sentences = [
    "I love cats",
    "I love dogs",
    "You love cats"
]

bigram_counts = defaultdict(Counter)

for sentence in sentences:
    words = sentence.split()

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]

        bigram_counts[current_word][next_word] += 1

counts = bigram_counts["love"]

total = sum(counts.values())


print(bigram_counts["love"])
print("-----------")

for word, count in counts.items():
    print(word, count/total)

