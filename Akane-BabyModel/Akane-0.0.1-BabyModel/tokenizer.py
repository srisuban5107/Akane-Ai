import json

class WordTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
        self.inverse_vocab = {}

    def build_vocab(self, lines):
        words = set()
        for line in lines:
            words.update(line.strip().split())
        idx = len(self.vocab)
        for w in words:
            if w not in self.vocab:
                self.vocab[w] = idx
                idx += 1
        self.inverse_vocab = {i:w for w,i in self.vocab.items()}

    def encode(self, text):
        tokens = [self.vocab["<BOS>"]]
        for w in text.strip().split():
            tokens.append(self.vocab.get(w, self.vocab["<UNK>"]))
        tokens.append(self.vocab["<EOS>"])
        return tokens

    def decode(self, ids):
        words = []
        for i in ids:
            w = self.inverse_vocab.get(i, "")
            if w not in ["<PAD>","<BOS>","<EOS>"]:
                words.append(w)
        return " ".join(words)
