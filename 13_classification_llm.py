import ollama

text = "I absolutely loved this movie."

prompt = f"""

classify the sentiment of the following text.

Possible labels:
-Positive
-Negative

Text:
{text}
return only the label
"""

response = ollama.generate(
    model= "qwen3:4b",
    prompt = prompt,
    options={
        "temperature": 0.0
    }
)
print(response["response"])