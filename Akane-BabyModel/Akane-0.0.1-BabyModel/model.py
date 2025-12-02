import torch
import torch.nn as nn
import torch.nn.functional as F

class BabyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=2, num_layers=2, seq_length=64):
        super(BabyTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)            # (batch, seq, embed)
        x = x.permute(1,0,2)                  # Transformer expects (seq, batch, embed)
        x = self.transformer(x)
        x = x.permute(1,0,2)
        logits = self.fc_out(x)
        return logits

    def generate(self, input_ids, max_length=50):
        self.eval()
        generated = input_ids.copy()
        for _ in range(max_length):
            x = torch.tensor([generated], dtype=torch.long)
            with torch.no_grad():
                logits = self.forward(x)
                next_token = torch.argmax(logits[0, -1]).item()
            generated.append(next_token)
            if next_token == 0:  # treat <PAD> or EOS as stopping
                break
        return generated
