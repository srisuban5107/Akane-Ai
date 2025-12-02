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
# Generate reply
# ----------------------
def generate_reply(prompt, max_len=50):
    ids = tokenizer.encode(prompt)
    ids = ids[:64]  # truncate if too long
    generated = model.generate(ids, max_length=max_len)
    return tokenizer.decode(generated)

# ----------------------
# Chat loop
# ----------------------
print("🌸 Akane is ready! Type 'exit' to quit.")
while True:
    user = input("You: ")
    if user.lower() == "exit":
        break
    reply = generate_reply(user)
    print("Akane:", reply)
