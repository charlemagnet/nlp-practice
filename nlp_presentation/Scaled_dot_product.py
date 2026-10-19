
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="cuda:0"
)
model.eval()

# 1. Gerçek token embedding'leri
sentence = "cat drinks milk"
ids = tokenizer(sentence, return_tensors="pt",
                add_special_tokens=False)["input_ids"].to("cuda")

tokens = tokenizer.convert_ids_to_tokens(ids[0])
print("TOKENS:", tokens)

with torch.no_grad():
    X = model.model.embed_tokens(ids)
    print("\nEMBEDDING SHAPE:", X.shape)
    print("First embedding:", X[0, 0, :5].float())

    # 2. Modelin öğrendiği Q, K, V
    layer = model.model.layers[0]
    attn = layer.self_attn
    hidden = layer.input_layernorm(X)

    seq_len = ids.shape[1]
    d = attn.head_dim

    Q = attn.q_proj(hidden).reshape(
        1, seq_len, model.config.num_attention_heads, d
    ).transpose(1, 2)

    K = attn.k_proj(hidden).reshape(
        1, seq_len, model.config.num_key_value_heads, d
    ).transpose(1, 2)

    V = attn.v_proj(hidden).reshape(
        1, seq_len, model.config.num_key_value_heads, d
    ).transpose(1, 2)

    Q = attn.q_norm(Q)
    K = attn.k_norm(K)

    # 3. RoPE (Position bilgisi)
    positions = torch.arange(seq_len, device="cuda").unsqueeze(0)
    cos, sin = model.model.rotary_emb(X, positions)
    Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

    # İlk Attention head, son tokenın Query'si
    query = Q[0, 0, -1]
    keys = K[0, 0]
    values = V[0, 0]

    print("\nQUERY:", query[:5].float())
    print("KEY (first token):", keys[0, :5].float())
    print("VALUE (first token):", values[0, :5].float())

    # 4. Query-Key Dot Product
    scores = query @ keys.T
    print("\n1. ATTENTION SCORES:", scores.float())

    # 5. Scaling
    scaled = scores / (d ** 0.5)
    print("2. SCALED SCORES:", scaled.float())

    # 6. Softmax
    weights = F.softmax(scaled.float(), dim=-1)

    print("\n3. ATTENTION WEIGHTS:")
    for token, weight in zip(tokens, weights):
        print(f"{token}: {weight.item():.2%}")

    # 7. Weighted Sum of Values
    output = weights @ values.float()
    print("\n4. FINAL ATTENTION OUTPUT:")
    print(output[:8])
    print("Output shape:", output.shape)
