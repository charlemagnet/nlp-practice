import ollama
import json
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

schema = {
    "type": "object",

    "properties": {

        "name": {
            "type": "string"
        },
        "age": {
            "type": "integer"
        },
        "city": {
            "type": "string"
        }
    },

    "required": [
        "name",
        "age",
        "city"
    ]
}

response = ollama.chat(

    model = "qwen3:4b",

    messages=[
        {
            "role": "user",
            "content": """

Extract information from this sentence.

John is 25 years old and lives in London.

Return as JSON.
"""
        }
    ],

    format=Person.model_json_schema(),

    think = False,

    options={
        "temperature": 0
    }
)

raw_output = response["message"]["content"]
print(raw_output)
data = json.loads(raw_output)

person = Person.model_validate_json(response["message"]["content"])
print(person.name)
print(person.age)
print(person.city)

print("\nName:", data["name"])
print("Age", data["age"])
print("City", data["city"])