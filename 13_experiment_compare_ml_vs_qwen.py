import json
import re
import time

import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
with open("sentiment_50.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_data = [item for item in data if item["split"] == "train"]
test_data = [item for item in data if item["split"] == "test"]

X_train_text = [item["text"] for item in train_data]
y_train = [item["label"] for item in train_data]

X_test_text = [item["text"] for item in test_data]
y_test = [item["label"] for item in test_data]

print(f"Train samples: {len(train_data)}")
print(f"Test samples : {len(test_data)}")


# ---------------------------------------------------------
# 2. Classical ML: TF-IDF + Logistic Regression
# ---------------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

ml_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

ml_train_start = time.perf_counter()

X_train = vectorizer.fit_transform(X_train_text)
ml_model.fit(X_train, y_train)

ml_train_time = time.perf_counter() - ml_train_start

ml_inference_start = time.perf_counter()

X_test = vectorizer.transform(X_test_text)
ml_predictions = ml_model.predict(X_test)

ml_inference_time = time.perf_counter() - ml_inference_start


# ---------------------------------------------------------
# 3. LLM: Qwen3:4b Zero-Shot Classification
# ---------------------------------------------------------
def classify_with_qwen(text):

    prompt = f"""
You are a sentiment classifier.

Classify the overall sentiment of this movie review.

Review:
{text}

Return the result as JSON.

/no_think
"""

    schema = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["POSITIVE", "NEGATIVE"]
            }
        },
        "required": ["label"]
    }

    response = ollama.chat(
        model="qwen3:4b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        think=False,

        format=schema,

        options={
            "temperature": 0.0,
            "num_predict": 30,
            "seed": 42
        }
    )

    raw_output = response["message"]["content"].strip()

    print("RAW:", repr(raw_output))

    try:
        result = json.loads(raw_output)

        label = result["label"].upper()

        if label in {"POSITIVE", "NEGATIVE"}:
            return label, raw_output

    except (json.JSONDecodeError, KeyError):
        pass

    return "UNKNOWN", raw_output


# Warm-up so model loading time is not counted
print("\nWarming up Qwen3:4b...")
_ = classify_with_qwen("This movie was good.")

llm_predictions = []

llm_inference_start = time.perf_counter()

for i, text in enumerate(X_test_text, start=1):
    prediction, raw_output = classify_with_qwen(text)
    llm_predictions.append(prediction)

    print(f"[{i:02d}/{len(X_test_text)}] Qwen -> {prediction}")

llm_inference_time = time.perf_counter() - llm_inference_start


# ---------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred, model_name):
    valid_indices = [
        i for i, pred in enumerate(y_pred)
        if pred in {"POSITIVE", "NEGATIVE"}
    ]

    valid_y_true = [y_true[i] for i in valid_indices]
    valid_y_pred = [y_pred[i] for i in valid_indices]

    return {
        "model": model_name,
        "accuracy": accuracy_score(valid_y_true, valid_y_pred),
        "precision": precision_score(
            valid_y_true,
            valid_y_pred,
            pos_label="POSITIVE",
            zero_division=0
        ),
        "recall": recall_score(
            valid_y_true,
            valid_y_pred,
            pos_label="POSITIVE",
            zero_division=0
        ),
        "f1": f1_score(
            valid_y_true,
            valid_y_pred,
            pos_label="POSITIVE",
            zero_division=0
        ),
        "unknown": len(y_pred) - len(valid_indices),
    }


ml_metrics = calculate_metrics(
    y_test,
    ml_predictions,
    "TF-IDF + Logistic Regression"
)

llm_metrics = calculate_metrics(
    y_test,
    llm_predictions,
    "Qwen3:4b Zero-Shot"
)


# ---------------------------------------------------------
# 5. Individual predictions
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("INDIVIDUAL TEST PREDICTIONS")
print("=" * 100)

for i, item in enumerate(test_data):
    print(f"\nID: {item['id']}")
    print(f"Text : {item['text']}")
    print(f"True : {item['label']}")
    print(f"ML   : {ml_predictions[i]}")
    print(f"Qwen : {llm_predictions[i]}")


# ---------------------------------------------------------
# 6. Final comparison
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("FINAL COMPARISON")
print("=" * 100)

print(f"""
{'MODEL':35} {'ACC':>8} {'PREC':>8} {'RECALL':>8} {'F1':>8} {'UNKNOWN':>8}
{'-' * 80}
{ml_metrics['model']:35} {ml_metrics['accuracy']:8.3f} {ml_metrics['precision']:8.3f} {ml_metrics['recall']:8.3f} {ml_metrics['f1']:8.3f} {ml_metrics['unknown']:8d}
{llm_metrics['model']:35} {llm_metrics['accuracy']:8.3f} {llm_metrics['precision']:8.3f} {llm_metrics['recall']:8.3f} {llm_metrics['f1']:8.3f} {llm_metrics['unknown']:8d}
""")


# ---------------------------------------------------------
# 7. Confusion matrices
# ---------------------------------------------------------
labels = ["NEGATIVE", "POSITIVE"]

print("\nML Confusion Matrix")
print("Rows = True labels, Columns = Predicted labels")
print(labels)
print(confusion_matrix(y_test, ml_predictions, labels=labels))

valid_llm_indices = [
    i for i, pred in enumerate(llm_predictions)
    if pred in labels
]

llm_true_valid = [y_test[i] for i in valid_llm_indices]
llm_pred_valid = [llm_predictions[i] for i in valid_llm_indices]

print("\nQwen Confusion Matrix")
print("Rows = True labels, Columns = Predicted labels")
print(labels)
print(confusion_matrix(llm_true_valid, llm_pred_valid, labels=labels))


# ---------------------------------------------------------
# 8. Classification reports
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("ML CLASSIFICATION REPORT")
print("=" * 100)
print(classification_report(
    y_test,
    ml_predictions,
    labels=labels,
    zero_division=0
))

print("\n" + "=" * 100)
print("QWEN CLASSIFICATION REPORT")
print("=" * 100)
print(classification_report(
    llm_true_valid,
    llm_pred_valid,
    labels=labels,
    zero_division=0
))


# ---------------------------------------------------------
# 9. Speed comparison
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("SPEED")
print("=" * 100)

print(f"Classical ML training time       : {ml_train_time:.4f} sec")
print(f"Classical ML test inference time : {ml_inference_time:.4f} sec")
print(f"Qwen test inference time         : {llm_inference_time:.4f} sec")

print(f"\nAverage ML time / test sample    : {ml_inference_time / len(test_data):.6f} sec")
print(f"Average Qwen time / test sample  : {llm_inference_time / len(test_data):.4f} sec")
