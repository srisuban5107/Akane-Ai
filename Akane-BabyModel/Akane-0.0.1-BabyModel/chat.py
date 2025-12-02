import os
import torch
from model import BabyTransformer
from save_utils import load_tokenizer

# ----------------------
# Paths inside same folder
# ----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "akane_model.pt")
tokenizer_path = os.path.join(script_dir, "akane_tokenizer.json")

# ----------------------
# Load tokenizer & model
# ----------------------
tokenizer = load_tokenizer(tokenizer_path)
vocab_size = len(tokenizer.vocab)

model = BabyTransformer(vocab_size)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ----------------------
# Lively generate function with temperature & top-k
# ----------------------
def generate_reply(prompt, max_len=50, temperature=1.0, top_k=5):
    ids = tokenizer.encode(prompt)
    ids = ids[:64]  # truncate if too long
    generated = ids.copy()
    
    for _ in range(max_len):
        x = torch.tensor([generated], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0, -1]  # last token
            logits = logits / temperature  # apply temperature
            probs = torch.softmax(logits, dim=-1)
            # top-k filtering
            topk_probs, topk_indices = torch.topk(probs, top_k)
            next_token = topk_indices[torch.multinomial(topk_probs, 1).item()].item()

        if next_token == tokenizer.vocab["<EOS>"] or next_token == tokenizer.vocab["<PAD>"]:
            break
        generated.append(next_token)
    
    return tokenizer.decode(generated)

# ----------------------
# Chat loop
# ----------------------
print("🌸 Akane is ready! Type 'exit' to quit.")
while True:
    user = input("You: ")
    if user.lower() == "exit":
        break
    reply = generate_reply(user, max_len=50, temperature=1.0, top_k=5)
    print("Akane:", reply)
