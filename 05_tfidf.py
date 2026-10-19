from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat is sleeping",
    "the dog is sleeping",
    "the bird is flying",
    "iburamincold is the best threatment"
]

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(documents)

words = vectorizer.get_feature_names_out()

for document, vector in zip(documents, x.toarray()):
    print("\n Document:")
    print(document)

    for word, score in zip(documents, x.toarray()):

        print("\nDocument:")
        print(document)

        for word, score in zip(words, vector):

            if score > 0:
                print(word, ":", round(score, 3))