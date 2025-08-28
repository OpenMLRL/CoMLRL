# Shuo: Tested on an A40, at least 40GB VRAM is required

import math
import re
from functools import partial

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer

# fmt: off
# Stopwords set for vocabulary analysis
STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "if", "because", "as", "what", "which",
    "this", "that", "these", "those", "then", "just", "so", "than", "such", "when",
    "who", "how", "where", "why", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing", "to",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "should", "now", "of"
}
# fmt: on


def proper_length_ratio_reward(
    completions1, completions2, target_min=2.0, target_max=3.0
):
    """Reward based on length ratio between completions (default: 2-3x)."""
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        len1, len2 = len(c1), len(c2)

        if len1 == 0:
            rewards.append(0.0)
            continue

        ratio = len2 / len1

        if target_min <= ratio <= target_max:
            reward = 1.0
        else:
            if ratio < target_min:
                distance = target_min - ratio
            else:
                distance = ratio - target_max

            reward = math.exp(-distance)

        rewards.append(float(reward))

    return rewards


def vocabulary_richness_reward(completions1, completions2):
    """Reward based on vocabulary richness improvement (Type-Token Ratio)."""

    def calculate_ttr(text, stopwords):
        """Calculate Type-Token Ratio excluding stopwords."""
        words = re.findall(r"\b\w+\b", text.lower())

        if stopwords:
            content_words = [word for word in words if word not in stopwords]
        else:
            content_words = words

        if not content_words:
            return 0.0

        types = len(set(content_words))
        tokens = len(content_words)

        return types / tokens if tokens > 0 else 0.0

    vocabulary_richness_reward.calculate_ttr = calculate_ttr
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        ttr1 = calculate_ttr(c1, STOPWORDS)
        ttr2 = calculate_ttr(c2, STOPWORDS)

        if ttr1 == 0:
            if ttr2 > 0:
                reward = 1.0
            else:
                reward = 0.0
        else:
            improvement = ttr2 / ttr1

            target_min = 1.2
            target_max = 2.0

            if improvement >= target_max:
                reward = 1.0
            elif improvement >= target_min:
                reward = (improvement - target_min) / (target_max - target_min)
            else:
                distance = target_min - improvement
                reward = math.exp(-2 * distance)

        rewards.append(float(reward))

    return rewards


def example_usage():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = MAGRPOConfig(
        output_dir="./magrpo_multi_reward_output",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        num_generations=8,
        max_new_tokens=256,
    )

    dataset_name = "trl-lib/tldr"
    dataset_split = "train[:100]"
    train_dataset = load_dataset(dataset_name, split=dataset_split)

    wandb_config = {
        "project": "mlrl",
        "entity": "nu-llpr",
        "name": "qwen-magrpo-multi-reward",
    }

    configured_proper_length_reward = partial(
        proper_length_ratio_reward, target_min=2, target_max=3
    )
    reward_funcs = [
        configured_proper_length_reward,
        vocabulary_richness_reward,
    ]
    reward_weights = [
        0.3,
        0.7,
    ]

    # fmt: off
    agents = []
    use_peft = False
    for _ in range(2):
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_peft:
            lora_config = LoraConfig(
                r=1024,
                lora_alpha=2048,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["embed_tokens", "lm_head"],
                fan_in_fan_out=False,
                task_type=TaskType.CAUSAL_LM,
            )
            lora_model = get_peft_model(base_model, lora_config)
            lora_model.print_trainable_parameters()
            agents.append(lora_model)
        else:
            agents.append(base_model)

    trainer = MAGRPOTrainer(
        agents=agents,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        wandb_config=wandb_config,
    )

    trainer.train()
    trainer.save_model(f"{config.output_dir}/final_models")
    print("Training complete!")


if __name__ == "__main__":
    example_usage()
