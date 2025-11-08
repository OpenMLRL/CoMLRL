import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.ippo import IPPOConfig, IPPOTrainer


def tldr_reward(prompts, responses) -> list[float]:
    """
    Reward Reddit TL;DR generations for matching a desired character length.

    We aim for ~220 characters and provide a smooth shaping signal where exact
    matches score near +1 and very short/long outputs trend toward -1.
    """

    target = 220
    scale = max(target / 2, 1)
    rewards = []

    for _prompt, response in zip(prompts, responses):
        trimmed = response.rstrip()
        char_len = len(trimmed)
        if char_len == 0:
            rewards.append(-1.0)
            continue

        shaped = 1.0 - abs(char_len - target) / scale
        rewards.append(float(max(-1.0, min(shaped, 1.0))))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on TL;DR summaries with PPO."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--critic-model", type=str, default=None)
    parser.add_argument("--separate-critic", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./ippo_tldr")
    parser.add_argument("--dataset-size", type=int, default=300)
    parser.add_argument("--num-train-epochs", type=int, default=12)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-6)
    parser.add_argument("--value-loss-coef", type=float, default=0.7)
    parser.add_argument("--wandb-project", type=str, default="ippo")
    parser.add_argument("--wandb-entity", type=str, default="OpenMLRL")
    parser.add_argument("--wandb-run-name", type=str, default="ippo_tldr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("trl-lib/tldr", split="train").select(
        range(args.dataset_size)
    )

    config = IPPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        use_separate_critic=args.separate_critic,
        critic_model_name_or_path=args.critic_model,
        learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        value_loss_coef=args.value_loss_coef,
    )

    wandb_config = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": args.wandb_run_name,
    }

    trainer = IPPOTrainer(
        model=args.model_name,
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
