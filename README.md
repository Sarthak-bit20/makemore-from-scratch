# Makemore — Character Level Language Models from Scratch

Implementation of character-level language models following 
Andrej Karpathy's "Neural Networks: Zero to Hero" series.
Built from scratch using Python and PyTorch.

## What is Makemore?

Makemore takes a dataset of words and learns to generate 
new similar words. Trained on 32,000 human names, it learns 
to generate new name-like words character by character.

## Implementations

### Part 1 — Bigram Model
`makemore-part1-divII.ipynb`
- Bigram character counting model
- 27x27 probability matrix
- Log likelihood loss
- Neural network bigram model
- Softmax, backpropagation, regularization

### Part 2 — MLP (Bengio 2003)
`makemore-part2-mlp-final.ipynb`
- Character embeddings (lookup table)
- 3 character context window
- MLP with tanh hidden layer
- Cross entropy loss
- Full training loop with gradient descent

## Architecture (Part 2)

Based on Bengio et al. 2003 paper:
"A Neural Probabilistic Language Model"
```
3 previous characters
→ embedding lookup (27x2 matrix C)
→ concatenate embeddings (6 dimensional)
→ hidden layer + tanh (6 → 100)
→ output layer + softmax (100 → 27)
→ probability of next character
```

## Setup
```bash
conda activate mlenv
jupyter lab
```

Requirements: Python 3.10, PyTorch, JupyterLab

## Dataset

`names.txt` — 32,000 human first names from ssa.gov

## Results

Part 1 Bigram NLL loss: ~2.45
Part 2 MLP NLL loss: ~2.1

## Learning Journey

This repo is part of my ML learning journey targeting 
EMAI (Erasmus Mundus Joint Master in AI) 2028.

Following Andrej Karpathy's Zero to Hero series to build 
deep understanding of language models from first principles.

## Reference

- [Karpathy's makemore](https://github.com/karpathy/makemore)
- [Bengio 2003 Paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
