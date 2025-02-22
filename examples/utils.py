import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

def train_model(
    model,
    train_loader,
    optimizer,
    scheduler=None,
    device='cuda',
    max_steps=10000,
    grad_clip=1.0,
    save_path=None,
    log_every=100
):
    """General training loop for any model"""
    model.train()
    progress_bar = tqdm(total=max_steps, desc="Training")
    step = 0
    running_loss = 0
    
    while step < max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= max_steps:
                break
                
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x, return_logits=True)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Update progress
            running_loss += loss.item()
            if (step + 1) % log_every == 0:
                avg_loss = running_loss / log_every
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                running_loss = 0
            
            step += 1
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Save model if path provided
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.7,
    top_k=40,
    device='cuda'
):
    """Generate text using any model with a generate method"""
    model.eval()
    with torch.no_grad():
        # Encode prompt
        x = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Generate
        generated = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        # Decode and return
        return tokenizer.decode(generated[0].tolist())

def setup_model(
    model_class,
    model_args,
    device='cuda',
    load_path=None,
):
    """Setup model with optional loading of weights"""
    model = model_class(**model_args).to(device)
    
    if load_path:
        print(f"Loading model from {load_path}...")
        model.load_state_dict(torch.load(load_path))
    
    return model

def setup_optimizer(
    model,
    learning_rate,
    weight_decay=0.0,
    scheduler_args=None
):
    """Setup optimizer and optional scheduler"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = None
    if scheduler_args and scheduler_args.get('type') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_args['max_steps'],
            eta_min=scheduler_args.get('min_lr', learning_rate/10)
        )
    
    return optimizer, scheduler 