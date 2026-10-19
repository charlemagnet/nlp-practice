import ollama

response = ollama.chat(
    model = "qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "Explain tokenization in one sentence"
        }
    ]
)

print(response["message"]["content"])