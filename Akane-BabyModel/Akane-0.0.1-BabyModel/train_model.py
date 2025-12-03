import os
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from model import BabyTransformer
from tokenizer import WordTokenizer
from dataset_loader import load_text_from_folder
from save_utils import save_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "Akane_dataset")
model_path = os.path.join(script_dir, "akane_model.pt")
tokenizer_path = os.path.join(script_dir, "akane_tokenizer.json")

# Load dataset
lines = load_text_from_folder(dataset_path)
print("Loaded lines:", len(lines))

# Tokenizer
tokenizer = WordTokenizer()
tokenizer.build_vocab(lines)
vocab_size = len(tokenizer.vocab)
print("Vocab size:", vocab_size)

# Encode lines
encoded = [torch.tensor(tokenizer.encode(line)) for line in lines]

# Pad sequences for mini-batches
def make_batch(seqs):
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    x = padded[:, :-1].to(device)
    y = padded[:, 1:].to(device)
    return x, y

# Model
model = BabyTransformer(vocab_size, embed_dim=128, num_heads=4, num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training
epochs = 15  # more epochs for small dataset
batch_size = 32

for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(encoded), batch_size):
        batch_seqs = encoded[i:i+batch_size]
        x, y = make_batch(batch_seqs)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

# Save
torch.save(model.state_dict(), model_path)
save_tokenizer(tokenizer, tokenizer_path)
print("Training complete!")
