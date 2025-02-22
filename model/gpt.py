import torch
from model.embeddings import TokenAndPositionalEmbedding
from model.block import TransformerBlock

class GPT(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=2048,
        dropout=0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embeddings = TokenAndPositionalEmbedding(vocab_size, d_model, max_seq_len)
        
        # Stack of transformer blocks
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.output = torch.nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, x, return_logits=False):
        # Truncate sequence if needed
        if x.size(1) > self.max_seq_len:
            x = x[:, :self.max_seq_len]
            
        # Get embeddings
        x = self.embeddings(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.output(x)
        
        if return_logits:
            return logits
        
        # Return softmax probabilities
        return torch.softmax(logits, dim=-1)
    
    def generate(self, start_tokens, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens given starting tokens"""
        self.eval()
        with torch.no_grad():
            tokens = start_tokens.clone()
            
            for _ in range(max_new_tokens):
                # Take last max_seq_len tokens if sequence is too long
                if tokens.size(1) > self.max_seq_len:
                    input_tokens = tokens[:, -self.max_seq_len:]
                else:
                    input_tokens = tokens
                
                # Get predictions for next token
                logits = self(input_tokens, return_logits=True)
                logits = logits[:, -1, :] / temperature  # Only care about last token
                
                # Optional top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append new token
                tokens = torch.cat([tokens, next_token], dim=1)
            
            return tokens
