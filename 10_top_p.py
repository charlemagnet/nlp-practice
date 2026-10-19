import ollama

prompt = "Continue this sentence: Artificial intelligence will"

for top_p in [0.2, 0.5, 0.9]:
    print("\n--------------")
    print("top_p", top_p)
    print("----------------")

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.8,
            "top_p": top_p
        }
    )

    print(response["message"]["content"])