import torch
from pathlib import Path
import sys

# Add root directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.utils import char_tokenizer

class ShakespeareTokenizer:
    def __init__(self):
        self.vocab, self.inv_vocab = char_tokenizer()
    
    def encode(self, text, return_tensors=None):
        """Encode text to token ids"""
        ids = [self.vocab.get(c, self.vocab['<|endoftext|>']) for c in text]
        if return_tensors == 'pt':
            return torch.tensor([ids], dtype=torch.long)
        return ids
    
    def decode(self, ids):
        """Decode token ids to text"""
        return ''.join(self.inv_vocab[id] for id in ids)
    
    @property
    def vocab_size(self):
        return len(self.vocab) 