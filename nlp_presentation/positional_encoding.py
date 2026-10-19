
import requests

MODEL = "qwen3:4b"

sentences = [
    "dog bites man",
    "man bites dog"
]

for sentence in sentences:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": (
                f"Sentence: {sentence}\n"
                "Who is biting whom? "
                "Answer in one short English sentence."
            ),
            "stream": False,
            "think": False,
            "options": {"temperature": 0}
        },
        timeout=180
    )
    response.raise_for_status()

    print("\nSentence:", sentence)
    print("Model answer:", response.json()["response"])
