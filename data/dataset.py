import torch
from torch.utils.data import Dataset
import requests
from data.utils import char_tokenizer, encode

class TextDataset(Dataset):
    def __init__(self, block_size: int = 128):
        self.block_size = block_size
        
        # Download tiny Shakespeare dataset
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        shakespeare_text = requests.get(url).text
        
        # Create vocabulary and encode text
        self.vocab, self.inv_vocab = char_tokenizer()
        data = torch.tensor(encode(shakespeare_text, self.vocab), dtype=torch.long)
        
        # Create training examples
        n = len(data)
        self.examples = []
        for i in range(0, n - block_size, block_size):
            self.examples.append(
                (data[i:i + block_size], data[i + 1:i + block_size + 1])
            )
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    @property
    def vocab_size(self):
        return len(self.vocab)
