import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

from nltk.tokenize import word_tokenize

text = "I love Natural Language Processing!"

tokens = word_tokenize(text)

print("Original text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nNumber of words with split()")
print(len(text.split()))

print("\nNumber of tokens")
print(len(tokens))