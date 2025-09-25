"""
MAGRPO training script for ArXiv abstract expansion.
Uses two agents to collaboratively generate expanded content from scientific abstracts.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
from comlrl.utils.reward_processor import RewardProcessors
import torch
from typing import Dict, Any


def background_agent_formatter(example: Dict[str, Any]) -> str:
    """Agent 1: Focuses on background and motivation."""
    abstract = example.get("abstract_text", "")
    return f"""Based on the following scientific abstract, expand content for an introduction section.

Abstract:
{abstract}

IMPORTANT INSTRUCTIONS:
- There is another agent that will provide methodology and implications
- You just need to focus on background and motivation
- Avoid repeating methodology and implications content
"""


def complementary_agent_formatter(example: Dict[str, Any]) -> str:
    """Agent 2: Focuses on methodology and implications."""
    abstract = example.get("abstract_text", "")
    return f"""Based on the following scientific abstract, expand content for an introduction section.

Abstract:
{abstract}

IMPORTANT INSTRUCTIONS:
- There is another agent that will provide the background and motivation
- You just need to focus on methodology and implications
- Avoid repeating background and motivation content
"""


def arxiv_combined_reward(completion1, completion2):
    """Calculate reward based on length ratio and diversity."""
    batch_size = len(completion1)
    rewards = []

    for i in range(batch_size):
        # Simple reward: length ratio + diversity bonus
        combined_text = completion1[i] + " " + completion2[i]
        length_score = min(len(combined_text.split()) / 100, 1.0)

        # Simple diversity check
        unique_words = len(set(combined_text.lower().split()))
        total_words = len(combined_text.split())
        diversity_score = unique_words / total_words if total_words > 0 else 0

        reward = (length_score + diversity_score) / 2
        rewards.append(reward)

    return rewards


def main():
    # Model configuration
    model_name = "Qwen/Qwen3-1.7B"

    # Load dataset
    dataset = load_dataset("LovelyBuggies/arXiv_abstract")
    train_dataset = dataset["train"].select(range(1000))
    eval_dataset = dataset["validation"].select(range(100))

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
        output_dir="./magrpo_arxiv_output",
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
        reward_func=arxiv_combined_reward,
        formatters=[background_agent_formatter, complementary_agent_formatter],
        args=magrpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        reward_processor=RewardProcessors.scale(factor=1),
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model("./magrpo_arxiv_output/final_model")
    print("Training completed and model saved!")


if __name__ == "__main__":
    main()
