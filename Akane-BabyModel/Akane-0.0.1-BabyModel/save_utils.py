# save_utils.py
import json
from tokenizer import WordTokenizer

def save_tokenizer(tokenizer, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.vocab, f, ensure_ascii=False, indent=2)

def load_tokenizer(path):
    t = WordTokenizer()
    import json
    with open(path, "r", encoding="utf-8") as f:
        t.vocab = json.load(f)
    t.inverse_vocab = {i:w for w,i in t.vocab.items()}
    return t

