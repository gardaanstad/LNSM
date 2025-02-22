import torch
import math

class TokenAndPositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Initialize position embeddings with sine/cosine
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.position_embedding.data.copy_(pe)
        
    def forward(self, x):
        # x is tensor of token indices (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Truncate or pad sequence if needed
        if seq_len > self.max_seq_len:
            x = x[:, :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # Get embeddings
        token_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        pos_emb = self.position_embedding[:, :seq_len, :]
        
        return token_emb + pos_emb
