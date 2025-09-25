"""
MAGRPO training script for TLDR summarization.
Uses two agents to collaboratively generate summaries with different perspectives.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
from comlrl.utils.reward_processor import RewardProcessors
import torch
from typing import Dict, Any


def summary_agent_formatter(example: Dict[str, Any]) -> str:
    """Agent 1: Creates concise summary."""
    prompt = example.get("prompt", "")
    return f"""Create a concise summary response to this post.

Query:
{prompt}

IMPORTANT INSTRUCTIONS:
- Provide a brief, focused summary in one sentence or a few sentences
- Be factual and informative
"""


def elaboration_agent_formatter(example: Dict[str, Any]) -> str:
    """Agent 2: Creates detailed summary with unique perspective."""
    prompt = example.get("prompt", "")
    return f"""Create a detailed summary response to this post.

Original Query:
{prompt}

IMPORTANT INSTRUCTIONS:
- Use more unique words
- Use some transition words to improve flow
"""


def tldr_combined_reward(completion1, completion2):
    """Calculate reward based on summary quality metrics."""
    batch_size = len(completion1)
    rewards = []

    for i in range(batch_size):
        combined_text = completion1[i] + " " + completion2[i]

        # Length-based reward (prefer moderate length)
        word_count = len(combined_text.split())
        if word_count < 20:
            length_score = 0.3
        elif word_count < 100:
            length_score = 0.8
        else:
            length_score = 0.6

        # Diversity reward
        unique_words = len(set(combined_text.lower().split()))
        total_words = len(combined_text.split())
        diversity_score = min(unique_words / max(total_words, 1), 1.0)

        # Combined reward
        reward = (length_score + diversity_score) / 2
        rewards.append(reward)

    return rewards


def main():
    # Model configuration
    model_name = "Qwen/Qwen3-1.7B"

    # Load dataset
    dataset = load_dataset("trl-lib/tldr")
    train_dataset = dataset["train"].select(range(1000))
    eval_dataset = dataset["test"].select(range(100))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Create agents
    agents = [
        AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        for _ in range(2)
    ]

    # Training configuration
    magrpo_args = MAGRPOConfig(
        output_dir="./magrpo_tldr_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=100,
        num_generations=4,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    # Create trainer
    trainer = MAGRPOTrainer(
        agents=agents,
        num_agents=2,
        reward_func=tldr_combined_reward,
        formatters=[summary_agent_formatter, elaboration_agent_formatter],
        args=magrpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        reward_processor=RewardProcessors.scale(factor=1),
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model("./magrpo_tldr_output/final_model")
    print("Training completed and model saved!")


if __name__ == "__main__":
    main()
