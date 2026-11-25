import spacy

nlp = spacy.load("en_core_web_sm")

text = """
Citrapps CTO Baris Kayirtar visited London on monday 
"""

doc = nlp(text)

for entity in doc.ents:
    print(
        entity.text,
        "->",
        entity.label_
    )