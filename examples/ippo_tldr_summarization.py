from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.ippo import IPPOConfig, IPPOTrainer


def tldr_reward(prompts, responses) -> list[float]:
    rewards = []
    for response in responses:
        # Preserve leading whitespace (models often prefix newline) but drop trailing noise.
        trimmed = response.rstrip()
        generation_length = len(trimmed)
        reward = -abs(generation_length - 200) / 50.0
        rewards.append(float(reward))
    return rewards


def build_prompt_formatter(tokenizer):
    def _formatter(example) -> str:
        if "prompt" not in example:
            raise KeyError("Expected 'prompt' field in dataset example.")

        prompt = example["prompt"]
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            messages = [
                {
                    "role": "system",
                    "content": "You summarize Reddit posts into concise TL;DRs.",
                },
                {"role": "user", "content": prompt},
            ]
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    return _formatter


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

    dataset = load_dataset("trl-lib/tldr", split="train").select(range(50))

    config = IPPOConfig(
        output_dir="./ippo_qwen_tldr",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        num_train_epochs=10,
        rollout_buffer_size=2,
        ppo_epochs=2,
        max_new_tokens=96,
        logging_steps=1,
        target_kl=0.5,
        temperature=0.7,
        top_p=0.6,
        do_sample=True,
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
        formatters=build_prompt_formatter(tokenizer),
        args=config,
        train_dataset=dataset,
        wandb_config=wandb_config,
        metrics_callback=rollout_metrics,
    )

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
