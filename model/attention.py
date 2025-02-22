import torch
import math

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        # Projections for queries, keys, values (all in one matrix)
        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        
        # For causal masking
        self.register_buffer("mask", None, persistent=False)

    def create_causal_mask(self, seq_len, device):
        """Create upper triangular mask for decoder attention"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

    def split_heads(self, x):
        """Reshape x to (batch_size, num_heads, seq_len, head_dim)"""
        batch_size, seq_len = x.size(0), x.size(1)
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        """Reverse of split_heads"""
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores + mask[:, :, :attn_scores.size(-2), :attn_scores.size(-1)]
        
        # Softmax to get probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values
        output = torch.matmul(attn_probs, v)
        return output, attn_probs

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Project input to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Split into heads
        q = self.split_heads(q)  # (batch_size, num_heads, seq_len, head_dim)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Create causal mask if not provided
        if mask is None:
            mask = self.create_causal_mask(seq_len, x.device)
        
        # Perform attention
        attn_output, attn_weights = self.scaled_dot_product_attention(
            q, k, v, mask=mask
        )
        
        # Combine heads
        attn_output = self.combine_heads(attn_output)
        
        # Final projection
        output = self.out_proj(attn_output)
        return output, attn_weights

# Test
if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    seq_len = 64
    batch_size = 4

    attn = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, weights = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")        # Should be [4, 64, 512]
    print(f"Weights shape: {weights.shape}")      # Should be [4, 8, 64, 64]
    
    # Verify causal masking
    assert torch.allclose(weights[0,0].tril(), weights[0,0]), "Mask not applied correctly!"