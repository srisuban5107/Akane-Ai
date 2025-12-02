import os
import torch
from torch import nn, optim
from model import BabyTransformer
from tokenizer import SimpleTokenizer
from dataset_loader import load_text_from_folder
from save_utils import save_tokenizer

# ----------------------
# Paths
# ----------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "Akane_dataset")
model_path = os.path.join(script_dir, "akane_model.pt")
tokenizer_path = os.path.join(script_dir, "akane_tokenizer.json")

# ----------------------
# Load dataset
# ----------------------
print("Looking for dataset at:", dataset_path)
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

lines = load_text_from_folder(dataset_path)
print("Loaded lines:", len(lines))

# ----------------------
# Build tokenizer
# ----------------------
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(lines)
vocab_size = len(tokenizer.vocab)
print("Vocab size:", vocab_size)

# ----------------------
# Create model
# ----------------------
model = BabyTransformer(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ----------------------
# Prepare training data (character-level)
# ----------------------
seq_len = 32
all_text = "".join(lines)
char_ids = tokenizer.encode(all_text)  # includes <BOS> and <EOS>

def get_batches(ids, seq_len):
    for i in range(0, len(ids)-seq_len, seq_len):
        x = ids[i:i+seq_len]
        y = ids[i+1:i+seq_len+1]
        yield torch.tensor([x]), torch.tensor([y])

# ----------------------
# Training loop
# ----------------------
epochs = 10  # small baby model
for epoch in range(epochs):
    total_loss = 0
    for x, y in get_batches(char_ids, seq_len):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

# ----------------------
# Save model and tokenizer INSIDE folder
# ----------------------
torch.save(model.state_dict(), model_path)
save_tokenizer(tokenizer, tokenizer_path)
print(f"Training done. Files saved inside folder:\n{model_path}\n{tokenizer_path}")
