import re
import json
import spacy
import ollama

from pydantic import BaseModel
from typing import Optional

# Loading tradional NER model

nlp = spacy.load("en_core_web_sm")

# Input text

text = """
John smith paid $1,250 to Acme Corp in London on May 12, 2026.
The laptop arrived with a broken screen, and he wants a full refund
"""

# Final result

result = {
    "person": None,
    "company": None,
    "location": None,
    "amount": None,
    "date": None,
    "problem": None,
    "requested_action": None
}

# STEP 1 - REGEX

money_pattern = r"\$[\d,]+(?:\.\d{2})?"

money_match = re.search(
    money_pattern,
    text
)

if money_match:
    result["amount"] = money_match.group()

# STEP 2 - TRADITIONAL NER

doc = nlp(text)

print("\nNER RESULTS:")

for entity in doc.ents:

    print(
        entity.text,
        "->",
        entity.label_
    )

    if entity.label_ == "PERSON" and result["person"] is None:
        result["person"] = entity.text

    elif entity.label_ == "ORG" and result["company"] is None:
        result["company"] = entity.text

    elif entity.label_ in {"GPE", "LOC"} and result["location"] is None:
        result["location"] = entity.text

    elif entity.label_ == "DATE" and result["date"] is None:
        result["date"] = entity.text


# STEP 3 - LLM FEEDBACK

class LLMExtraction(BaseModel):
    problem: Optional[str] = None
    requested_action: Optional[str] = None


prompt = f"""
Extract customer complaint information from the text.

Fields:

problem:
The product issue or problem experienced by the customer.

requested_action:
What the customer wants the company to do to solve the problem.

Rules:
- Extract information only from the given text.
- Do not invent information.
- Use null only if the information is truly not present.
- Keep the extracted values short.

Text:
{text}
"""

response = ollama.chat(

    model="qwen3:4b",

    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],

    format=LLMExtraction.model_json_schema(),

    think=False,

    options={
        "temperature": 0
    }
)


llm_result = LLMExtraction.model_validate_json(
    response["message"]["content"]
)


result["problem"] = llm_result.problem

result["requested_action"] = (
    llm_result.requested_action
)

# STEP 4 — FINAL RESULT

print("\nFINAL HYBRID EXTRACTION:")

print(
    json.dumps(
        result,
        indent=4
    )
)