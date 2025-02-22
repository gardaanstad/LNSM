"""Configuration parameters for the GPT model"""

# Model architecture
VOCAB_SIZE = 128    # ASCII characters + special tokens
D_MODEL = 256      # Embedding dimension
NUM_LAYERS = 6     # Number of transformer blocks
NUM_HEADS = 8      # Number of attention heads
MAX_SEQ_LEN = 128  # Maximum sequence length
DROPOUT = 0.1      # Dropout rate

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 1000
MAX_STEPS = 10000

# Generation parameters
TEMPERATURE = 0.7
TOP_K = 40
MAX_NEW_TOKENS = 100
