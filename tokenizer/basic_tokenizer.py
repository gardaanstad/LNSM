class BasicTokenizer:
    def __init__(self):
        self.vocab = {}  # {'<|endoftext|>': 0, 'the': 1, ...}
        self.merges = {}
    
    def train(self, text, vocab_size):
        words = text.split()
        vocab = sorted(set(words))
        self.vocab = {word: i for i, word in enumerate(vocab[:vocab_size])}
    
    def encode(self, text):
        return [self.vocab[word] for word in text.split() if word in self.vocab]
    
    def decode(self, tokens):
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join([inv_vocab.get(t, '') for t in tokens])