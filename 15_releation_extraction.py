import ollama
import json

schema = {
    "type": "object",

    "properities": {

        "person": {
            "type": "string"
        },

        "job_title": {
            "type": "string"
        },

        "company": {
            "type": "string"
        },

        "location": {
            "type": "string"
        }
    },

    "required": [
        "person",
        "job_title",
        "company",
        "location"
    ]
}

text = """
The CTO of Citrapps, Baris Kayirtar
lives in California.
"""

response = ollama.chat(
    model = "qwen3:4b",

    messages=[
        {
            "role": "user",
            "content": f"""
Extract the information
from the following text.

Text:
{text}
"""
        }
    ],

    format=schema,
    think=False,

    options={
        "temperature": 0
    }
)

result = json.loads(
    response["message"]["content"]
)

print(result)