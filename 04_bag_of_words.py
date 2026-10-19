from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I like cats",
    "I like dogs",
    "Cats and dogs are animals"
]

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(documents)

print("Vocubulary:")
print(vectorizer.get_feature_names_out())

print("\nBag of Words Matrix")
print(x.toarray())