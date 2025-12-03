# model.py
import torch
import torch.nn as nn

class BabyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, max_positions=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # make position embedding large enough (default 512). Avoids IndexError.
        self.position_embedding = nn.Embedding(max_positions, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        batch, seq_len = x.shape
        tok_emb = self.token_embedding(x)                       # (batch, seq, embed)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # clamp to available positions to be safe
        max_pos = self.position_embedding.num_embeddings
        pos_ids = pos_ids.clamp(max=max_pos - 1)
        pos_emb = self.position_embedding(pos_ids)              # (1, seq, embed)
        x = tok_emb + pos_emb
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits
