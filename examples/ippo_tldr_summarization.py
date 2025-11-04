from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.ippo import IPPOConfig, IPPOTrainer


def tldr_reward(prompts, responses, target_words: int = 60) -> list[float]:
    rewards = []
    for prompt, response in zip(prompts, responses):
        score = 0.0
        normalized = response.strip()

        if normalized.lower().startswith("tl;dr"):
            score += 0.5
        else:
            score -= 0.1

        word_count = max(len(normalized.split()), 1)
        length_ratio = min(word_count / target_words, target_words / word_count)
        score += 0.5 * length_ratio

        if any(
            keyword in normalized.lower()
            for keyword in ["summary", "overall", "in short"]
        ):
            score += 0.1

        rewards.append(float(score))
    return rewards


def prompt_formatter(example) -> str:
    if "prompt" in example:
        return example["prompt"]
    raise KeyError("Expected 'prompt' field in dataset example.")


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("trl-lib/tldr", split="train").select(range(20))

    config = IPPOConfig(
        output_dir="./ippo_qwen_tldr",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        num_train_epochs=5,
        rollout_buffer_size=2,
        ppo_epochs=2,
        max_new_tokens=128,
        logging_steps=1,
    )

    wandb_config = {
        "entity": "nu-llpr",
        "project": "mlrl",
        "name": "ippo_tldr",
    }

    trainer = IPPOTrainer(
        model=model_name,
        tokenizer=tokenizer,
        reward_func=tldr_reward,
        formatters=prompt_formatter,
        args=config,
        train_dataset=dataset,
        wandb_config=wandb_config,
    )

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
