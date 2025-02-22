import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Add root directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.gpt import GPT
from data.dataset import TextDataset
from examples.utils import train_model, setup_model, setup_optimizer, generate_text
from examples.shakespeare.tokenizer import ShakespeareTokenizer
import config

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and tokenizer
    dataset = TextDataset(block_size=config.MAX_SEQ_LEN)
    tokenizer = ShakespeareTokenizer()
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    # Setup model
    model_args = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': config.D_MODEL,
        'num_layers': config.NUM_LAYERS,
        'num_heads': config.NUM_HEADS,
        'max_seq_len': config.MAX_SEQ_LEN,
        'dropout': config.DROPOUT
    }
    model = setup_model(GPT, model_args, device=device)
    
    # Setup optimizer and scheduler
    scheduler_args = {
        'type': 'cosine',
        'max_steps': config.MAX_STEPS,
        'min_lr': config.LEARNING_RATE/10
    }
    optimizer, scheduler = setup_optimizer(
        model,
        config.LEARNING_RATE,
        config.WEIGHT_DECAY,
        scheduler_args
    )
    
    # Train
    print("Starting training...")
    train_model(
        model,
        train_loader,
        optimizer,
        scheduler,
        device=device,
        max_steps=config.MAX_STEPS,
        save_path="shakespeare_gpt.pt"
    )
    
    # Generate sample
    print("\nGenerating sample text...")
    sample_text = generate_text(
        model,
        tokenizer,
        prompt="ROMEO:",
        max_new_tokens=100,
        temperature=config.TEMPERATURE,
        top_k=config.TOP_K,
        device=device
    )
    print(f"\nGenerated text:\n{sample_text}")

if __name__ == "__main__":
    main() 