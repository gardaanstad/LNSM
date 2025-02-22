import torch
from model.attention import MultiHeadAttention

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Apply attention and residual connection
        attn_output, _ = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Apply feed-forward and residual connection
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x