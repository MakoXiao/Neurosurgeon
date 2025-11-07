# RL-Based Collaborative Inference

This project implements a reinforcement learning-based collaborative inference framework that combines model partitioning and pruning compression.

## Features

- **RL-based partition point selection**: Uses PPO algorithm to dynamically select optimal partition points
- **Pruning compression**: Structured/unstructured pruning for intermediate feature compression
- **Hybrid action space**: Discrete partition point + continuous compression rate
- **Accuracy-latency trade-off**: Optimizes both accuracy and latency simultaneously

## Installation

Make sure you're using the `neurosurgeon_env` virtual environment:

```bash
source ../neurosurgeon_env/bin/activate
```

Install required packages:

```bash
pip install matplotlib seaborn tqdm
```

## Usage

### Training

```bash
python train.py \
    --data_dir ../data/caltech-101 \
    --output_dir ./results \
    --max_steps 10000 \
    --network_bandwidth 10.0 \
    --pruning_type structured \
    --use_cuda
```

### Evaluation

```bash
python evaluate.py \
    --data_dir ../data/caltech-101 \
    --model_path ./results/train_XXX/final_model.pt \
    --output_dir ./experiments \
    --network_bandwidth 10.0 \
    --use_cuda
```

## Directory Structure

```
rl_collaborative_inference/
├── src/                    # Source code
│   ├── actor_critic.py    # Actor-Critic networks
│   ├── env.py             # RL environment
│   ├── ppo.py             # PPO algorithm
│   ├── pruning.py         # Pruning modules
│   ├── model_partition.py # Model partitioning
│   ├── state_space.py     # State space definition
│   └── dataset_loader.py # Dataset loader
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── results/               # Training results
└── experiments/           # Evaluation results and plots
```

## Results

The evaluation script generates:
- `comparison.png`: Bar charts comparing accuracy and latency
- `tradeoff.png`: Scatter plot showing accuracy-latency trade-off
- `results.json`: Detailed numerical results

