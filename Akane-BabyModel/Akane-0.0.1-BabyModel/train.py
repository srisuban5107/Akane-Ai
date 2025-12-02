from dataset_loader import load_text_from_folder
from tokenizer import SimpleTokenizer
from model import BabyTransformer
import torch

# Load data
lines = load_text_from_folder("Akane_dataset")

# Debug
print("Loaded lines:", len(lines))

# Build tokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(lines)

# Initialize model
vocab_size = len(tokenizer.vocab)
model = BabyTransformer(vocab_size)


