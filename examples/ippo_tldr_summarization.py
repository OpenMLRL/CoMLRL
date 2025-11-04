from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.ippo import IPPOConfig, IPPOTrainer


def tldr_reward(prompts, responses) -> list[float]:
    rewards = []
    for response in responses:
        normalized = response.strip()
        generation_length = len(normalized)
        reward = -abs(generation_length - 200) / 50.0
        rewards.append(float(reward))
    return rewards


def prompt_formatter(example) -> str:
    if "prompt" in example:
        return example["prompt"]
    raise KeyError("Expected 'prompt' field in dataset example.")


def rollout_metrics(rollouts):
    if not rollouts:
        return {}

    char_lengths = [sample.metadata.get("char_length", 0.0) for sample in rollouts]
    return {
        "response_char_length_mean": float(sum(char_lengths) / len(char_lengths)),
    }


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
        target_kl=0.5,
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
        metrics_callback=rollout_metrics,
    )

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
