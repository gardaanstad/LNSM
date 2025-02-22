# Lille norske språkmodell

This is my attempt to implement a fully functional Generative Pretrained Transformer (GPT) model from scratch in Python using PyTorch.

Example usage:

```bash
python examples/shakespeare/train.py
```

```bash
python examples/shakespeare/generate.py --prompt "HAMLET:" --temperature 0.9 --max_tokens 300
```


Project structure:
- [data](data): Parsing text and creating datasets to train on
- [model](model): Creating the model
- [training](training): Training the model
- [tokenizer](tokenizer): Tokenization
- [config.py](files/config.py): Hyperparameters