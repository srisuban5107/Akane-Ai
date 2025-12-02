class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }
        self.inverse_vocab = {}

    def build_vocab(self, lines):
        unique_chars = set("".join(lines))
        idx = len(self.vocab)

        for ch in unique_chars:
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1

        self.inverse_vocab = {i: ch for ch, i in self.vocab.items()}

    def encode(self, text):
        encoded = [self.vocab["<BOS>"]]
        for ch in text:
            encoded.append(self.vocab.get(ch, self.vocab["<UNK>"]))
        encoded.append(self.vocab["<EOS>"])
        return encoded

    def decode(self, ids):
        text = []
        for i in ids:
            if i in self.inverse_vocab:
                ch = self.inverse_vocab[i]
                if ch not in ["<PAD>", "<BOS>", "<EOS>"]:
                    text.append(ch)
        return "".join(text)
