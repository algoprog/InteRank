# InteRank: Reasoning-Intensive Document Ranking with Small Language Models
[![arxiv](https://img.shields.io/badge/arXiv-2504.03947-b31b1b.svg)](https://arxiv.org/abs/2504.03947)

InteRank is a novel approach for training compact language models (< 3B parameters) to perform reasoning-intensive document ranking with performance comparable to models over 20x larger. Our methodology combines knowledge distillation from a large teacher model with reinforcement learning optimization to create efficient yet powerful ranking models that can explain their decisions.

## Overview

Key features:
- Achieves state-of-the-art performance on the BRIGHT benchmark using only a 3B parameter model
- Generates natural language explanations to justify ranking decisions
- Uses a two-stage training approach combining knowledge distillation and reinforcement learning
- Requires no human annotations for training
- Supports both query generation and document relevance assessment

## Training Process

1. **Synthetic Data Generation**: Automatically generates training data from StackExchange question-answer pairs using a large teacher model (Llama 3.3 70B)

2. **Knowledge Distillation**: Transfers initial reasoning capabilities from the teacher to a compact student model (Llama 3.2 3B)

3. **Reinforcement Learning**: Refines reasoning capabilities by rewarding high-quality explanations and accurate relevance predictions

## Model Architecture

- Base Model: Llama 3.2 3B
- Training: QLoRA with 4-bit quantization and rank-64 adapters
- Context Length: 4K tokens
- Hardware Requirements: Single A100 GPU

## Results

Our 3B parameter model achieves:
- 27.4% average nDCG@10 across all domains on BRIGHT benchmark
- 3rd place on BRIGHT leaderboard
- Outperforms recent approaches like Reason-to-Rank (nDCG@5 26.2 vs 19.6)
- Performance comparable to ensemble models using 70B+ parameters
