# Recreating GPT 2(124M)

# Table of Contents
- [Overview](#overview) 
- [Files](#files)
- [Requirements](#requirements)
- [Installation and Usage](#installation-and-usage)


# Overview
This project demonstrates how to recreate the GPT-2 124M model using custom scripts for data preparation and model training. We use the FineWeb-Edu dataset for training and validation, tokenize it using the GPT-2 tokenizer, and train a model from scratch.

- Dataset: FineWeb-Edu (10B tokens).
- Tokenizer: GPT-2 tokenizer via tiktoken.
- Model: GPT-2 architecture, 124M parameters.
- Sharding: Data is split into manageable shards for efficient processing.
- Training: Using PyTorch, with support for distributed training (multi-GPU).

# Files
1. fineweb.py:
    - Downloads the FineWeb-Edu dataset.
    - Tokenizes the dataset using the GPT-2 tokenizer.
    - Splits the tokenized data into shards (100 million tokens per shard) and saves them to disk.
2. train_gpt2.py:
    - Loads the tokenized shards from disk.
    - Initializes and trains the GPT-2 124M model.
    - Supports distributed training (multi-GPU with DDP) and checkpointing.
  
# Requirements
```
pip install torch datasets tiktoken tqdm
```

# Installation and Usage
1. Clone the Repository
2. Run fineweb.py to download, tokenize, and shard the dataset.
```
python fineweb.py
```
This will create a directory edu_fineweb10B containing .npy files (tokenized shards).
3. Train the GPT-2 Model
```
python train_gpt2.py
```
This script will automatically load the shards, train the model, and log the progress. If using multiple GPUs, use the following command to run with Distributed Data Parallel (DDP):
```
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```
4. Checkpoints and Logs
    - Checkpoints will be saved periodically to the log directory.
    - Validation losses and model accuracy will be logged as training progresses.

