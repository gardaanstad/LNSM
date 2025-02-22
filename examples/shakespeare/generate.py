import torch
import argparse
from pathlib import Path
import sys

# Add root directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.gpt import GPT
from examples.utils import setup_model, generate_text
from examples.shakespeare.tokenizer import ShakespeareTokenizer
import config

def main():
    parser = argparse.ArgumentParser(description='Generate Shakespeare-style text using trained GPT model')
    parser.add_argument('--model_path', type=str, default='examples/shakespeare/shakespeare_gpt.pt',
                      help='Path to the trained model')
    parser.add_argument('--prompt', type=str, default='ROMEO:',
                      help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=40,
                      help='Top K sampling parameter (0 to disable)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create tokenizer
    tokenizer = ShakespeareTokenizer()
    
    # Setup model
    model_args = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': config.D_MODEL,
        'num_layers': config.NUM_LAYERS,
        'num_heads': config.NUM_HEADS,
        'max_seq_len': config.MAX_SEQ_LEN,
        'dropout': config.DROPOUT
    }
    model = setup_model(GPT, model_args, device=args.device, load_path=args.model_path)
    
    # Generate text
    print(f"\nGenerating text from prompt: {args.prompt}")
    print("-" * 50)
    
    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
    
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main() 