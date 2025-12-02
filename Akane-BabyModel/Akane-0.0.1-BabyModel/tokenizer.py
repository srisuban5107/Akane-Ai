from tokenizer import Tokenizer, models, trainers, pre_tokenizers, processors

class SimpleTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.trainer = trainers.BpeTrainer(special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

    def build_vocab(self, lines):
        self.tokenizer.train_from_iterator(lines, trainer=self.trainer)
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="<BOS> $A <EOS>",
            pair="<BOS> $A <EOS> <BOS> $B <EOS>",
            special_tokens=[
                ("<BOS>", self.tokenizer.token_to_id("<BOS>")),
                ("<EOS>", self.tokenizer.token_to_id("<EOS>"))
            ]
        )

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
