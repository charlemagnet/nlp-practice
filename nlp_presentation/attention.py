import torch
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager"
)

sentence = "The cat drank the milk because it was thirsty."

inputs = tokenizer(
    sentence,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True
    )

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("\nTokens:")
print(tokens)

# Last layer
attention = outputs.attentions[-1][0]

print("\nAttention shape:")
print(attention.shape)

it_index = tokens.index("it")

# Her attention head'i ayrı ayrı göster
for head in range(attention.shape[0]):

    weights = attention[head, it_index]

    print(f"\n--- HEAD {head + 1} ---")

    top_values, top_indices = torch.topk(weights, k=4)

    for value, index in zip(top_values, top_indices):
        print(
            f"{tokens[index]:12} -> {value.item():.4f}"
        )
cat_index = tokens.index("cat")

print("\nAttention from 'it' to 'cat' in each head:")

for head in range(attention.shape[0]):

    score = attention[head, it_index, cat_index]

    print(
        f"Head {head + 1}: {score.item():.4f}"
    )