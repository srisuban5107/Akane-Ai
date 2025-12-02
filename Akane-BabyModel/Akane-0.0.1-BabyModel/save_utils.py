import json

def save_tokenizer(tokenizer, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=2)

def load_tokenizer(path):
    import tokenizer as tok
    import json
    t = tok.SimpleTokenizer()
    with open(path, "r", encoding="utf-8") as f:
        t.vocab = json.load(f)
    t.inverse_vocab = {i: ch for ch, i in t.vocab.items()}
    return t
