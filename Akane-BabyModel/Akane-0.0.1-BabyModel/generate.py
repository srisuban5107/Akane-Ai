import os
import torch
import random
from model import BabyTransformer
from save_utils import load_tokenizer

# -------------------------
# Paths
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "akane_model.pt")
tokenizer_path = os.path.join(script_dir, "akane_tokenizer.json")

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Load tokenizer
# -------------------------
tokenizer = load_tokenizer(tokenizer_path)
vocab_size = len(tokenizer.vocab)

# -------------------------
# Load model
# -------------------------
model = BabyTransformer(vocab_size, embed_dim=128, num_heads=4, num_layers=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------
# Generate reply function
# -------------------------
def generate_reply(prompt, max_len=12, temperature=0.8, top_k=10):
    """
    Generate a short reply using the trained BabyTransformer.
    - Always returns a reply, even if the model outputs <UNK> or fails.
    - max_len: maximum number of tokens in reply.
    - temperature: randomness control.
    - top_k: top-k sampling.
    """
    try:
        # Encode prompt tokens
        ids = tokenizer.encode(prompt)
        ids = ids[:64]  # limit input length
        generated = ids.copy()

        for _ in range(max_len):
            x = torch.tensor([generated], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(x)[0, -1] / max(temperature, 1e-8)
                probs = torch.softmax(logits, dim=-1)
                k = min(top_k, probs.size(0))
                topk_probs, topk_idx = torch.topk(probs, k)
                next_token = topk_idx[torch.multinomial(topk_probs, 1).item()].item()

            # Stop on PAD or EOS
            if next_token in [tokenizer.vocab.get("<PAD>"), tokenizer.vocab.get("<EOS>")]:
                break

            generated.append(next_token)

        # Decode only new tokens
        new_ids = generated[len(ids):]
        decoded = tokenizer.decode(new_ids).strip()

        # If decoding fails or gives <UNK>, pick a random word from vocab
        if not decoded or "<UNK>" in decoded or len(decoded) < 2:
            random_ids = [random.randint(0, vocab_size - 1) for _ in range(random.randint(1, 3))]
            decoded = tokenizer.decode(random_ids).strip()

        # Limit length to first sentence for mini-chat feel
        if "." in decoded:
            decoded = decoded.split(".")[0]

        return decoded

    except Exception as e:
        # Safety fallback
        random_ids = [random.randint(0, vocab_size - 1) for _ in range(random.randint(1, 3))]
        return tokenizer.decode(random_ids).strip()
