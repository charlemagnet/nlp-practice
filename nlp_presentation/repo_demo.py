
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="cuda:0"
)
model.eval()

# 1. Gerçek token embedding
token_ids = tokenizer(
    "dog", add_special_tokens=False, return_tensors="pt"
)["input_ids"].to("cuda")

with torch.no_grad():
    X = model.model.embed_tokens(token_ids)

    print("TOKEN IDS:", token_ids.tolist())
    print("EMBEDDING SHAPE:", X.shape)
    print("EMBEDDING:", X[0, 0, :8].float().cpu().numpy())

    # 2. İlk Transformer katmanından Query ve Key
    attn = model.model.layers[0].self_attn
    hidden = model.model.layers[0].input_layernorm(X)

    Q = attn.q_proj(hidden)
    K = attn.k_proj(hidden)

    Q = Q.view(1, 1, model.config.num_attention_heads, attn.head_dim)
    K = K.view(1, 1, model.config.num_key_value_heads, attn.head_dim)

    Q = attn.q_norm(Q).transpose(1, 2)
    K = attn.k_norm(K).transpose(1, 2)

    print("\nQUERY BEFORE ROPE:", Q[0, 0, 0, :8].float().cpu())
    print("KEY BEFORE ROPE:", K[0, 0, 0, :8].float().cpu())

    # 3. RoPE: Aynı token, farklı pozisyonlar
    for position in [0, 2]:
        position_ids = torch.tensor([[position]], device="cuda")

        cos, sin = model.model.rotary_emb(X, position_ids)
        Q_rot, K_rot = apply_rotary_pos_emb(Q, K, cos, sin)

        print(f"\n--- POSITION {position} ---")
        print("Query:", Q_rot[0, 0, 0, :8].float().cpu())
        print("Key:", K_rot[0, 0, 0, :8].float().cpu())

print("\nDevice:", X.device)
