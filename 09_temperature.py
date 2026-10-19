import ollama

prompt = "Give me a creative name for coffee shop."

for temperature in [0.1, 0.7, 1.2]:

    print("\n--------------")
    print("Temperature:", temperature)
    print("----------------")

    response = ollama.chat(
        model = "qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": temperature
        }
    )

    print(response["message"]["content"])