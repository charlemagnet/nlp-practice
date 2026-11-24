from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Train data
texts = [
    "I love this movie",
    "This movie is amazing",
    "I realy enjoyed the film",
    "This movie is terrible",
    "I hate this film",
    "This was a waste of time"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negarive",
    "negative"
]

# 1. Text -> TF-IDF vectors
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# 2. Train classifier
model = LogisticRegression()

model.fit(X, labels)

# 3. New text
new_text = [
    "I realy liked this movie"
]

# Important:
# Trasnform(), not fit_transform()

X_new = vectorizer.transform(new_text)

# 4. Predict
prediction = model.predict(X_new)

print("Prediction:", prediction[0])
