# How to Use MAGRPOTrainer

A quick start guide for researchers to run MAGRPO experiments with custom data and reward functions.

## What You Need

- **Models**: 2+ language models (same or different)
- **Reward Function**: Function that evaluates agent completions
- **Dataset**: Your training data
- **GPU**: Required for training (24GB+ VRAM recommended)

## Quick Start

### 1. Install and Setup

```bash
cd CoMLRL
pip install -e .
wandb login  # Optional: for experiment tracking
```

### 2. Create Your Reward Function

Your reward function must accept completions from all agents and return a list of scores:

```python
def my_reward_function(completions1, completions2, batch_items=None):
    """Reward function that compares completions from 2 agents."""
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Your reward logic here
        # Example: length-based reward
        if len(c2) > len(c1) * 2:
            reward = 1.0
        else:
            reward = 0.5
        rewards.append(reward)
    return rewards
```

### 2.1 Reward Processors (Optional)

Reward processors transform your raw rewards before training. Common use cases:

```python
from comlrl.utils.reward_processor import RewardProcessors

# Scale rewards (amplify small rewards for better training signals)
RewardProcessors.scale(factor=100.0)  # 0.5 → 50.0

# Clamp rewards to prevent extreme values
RewardProcessors.clamp(min_val=-10.0, max_val=10.0)

# Apply sigmoid scaling
RewardProcessors.sigmoid_scale()

# Identity (no change)
RewardProcessors.identity()
```

**Why use reward processors?**
- **Scale**: Policy gradients work better with larger reward magnitudes
- **Clamp**: Prevent extreme values that could destabilize training
- **Sigmoid**: Convert any range to 0-1 range

### 3. Prepare Your Dataset

Your dataset should be a list of dictionaries with at least a "prompt" field:

```python
from datasets import Dataset

train_data = {
    "prompt": [
        "Write a story about a robot:",
        "Explain quantum physics:",
        "Create a recipe for chocolate cake:",
    ]
}
train_dataset = Dataset.from_dict(train_data)
```

### 4. Configure Training

```python
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
agents = [AutoModelForCausalLM.from_pretrained(model_name) for _ in range(2)]

# Training configuration
config = MAGRPOConfig(
    output_dir="./my_experiment",
    num_train_epochs=3,
    learning_rate=5e-5,
    num_generations=4,  # Samples per agent per prompt
    max_new_tokens=128,
    per_device_train_batch_size=1,  # Must be 1 (see explanation below)
)
```

### 5. Create and Train

```python
# Create trainer
trainer = MAGRPOTrainer(
    agents=agents,
    reward_funcs=my_reward_function,
    reward_processors=RewardProcessors.scale(factor=100.0),  # Scale rewards
    args=config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    wandb_config={
        "project": "my-project",
        "entity": "my-username",
        "name": "my-experiment",
    },
)

# Train
trainer.train()

# Save models
trainer.save_model("./my_experiment/final_models")
```

## Key Parameters

### Required
- `agents`: List of model objects
- `reward_funcs`: Your reward function(s)
- `args`: MAGRPOConfig with training settings
- `train_dataset`: Your training data

### Important Config Options
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate (typically 1e-5 to 5e-5)
- `num_generations`: Samples per agent per prompt (4-8 recommended)
- `max_new_tokens`: Maximum completion length
- `num_agents`: Number of agents (default: 2)
- `per_device_train_batch_size`: **Must be 1** - MAGRPO processes one prompt at a time because reward functions evaluate completions for a single input prompt

### Optional
- `reward_processors`: Transform rewards (e.g., `RewardProcessors.scale(factor=100.0)`)
- `reward_weights`: Weight multiple reward functions (e.g., `[0.7, 0.3]`)
- `wandb_config`: Experiment tracking
- `eval_dataset`: Evaluation data

## Next Steps

- Experiment with different reward functions
- Try multi-turn training (`num_turns > 1`)
- Use heterogeneous agents (different models)
- Explore reward combinations with multiple reward functions
