import numpy as np

Q = np.array([[1.0, 0.0]])

K = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

V = np.array([
    [10.0, 2.0],
    [1.0, 8.0]
])

scores = Q @ K.T

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

weights = softmax(scores[0])

output = weights @ V

print("Scores:", scores)
print("Weights:", weights)
print("Output:", output)