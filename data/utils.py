import torch
import numpy as np
from typing import List

def char_tokenizer():
    """Simple character-level tokenizer"""
    chars = [chr(i) for i in range(128)]  # ASCII characters
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab['<|endoftext|>'] = len(vocab)
    return vocab, {v: k for k, v in vocab.items()}

def encode(text: str, vocab: dict) -> List[int]:
    """Encode text to token ids"""
    return [vocab.get(c, vocab['<|endoftext|>']) for c in text]

def decode(ids: List[int], inv_vocab: dict) -> str:
    """Decode token ids to text"""
    return ''.join(inv_vocab[id] for id in ids)

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str = 'cpu'):
    """Get a random batch of data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
