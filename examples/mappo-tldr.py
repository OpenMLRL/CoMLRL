import argparse
from functools import partial

from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer


def dual_length_reward(
    short_responses: list[str],
    long_responses: list[str],
    ratio_min: float = 2.0,
    ratio_max: float = 3.0,
    short_target: int = 220,
    short_scale: float | None = None,
) -> list[float]:
    """
    Reward two agents so that the first stays near a target length while the second is 2-3x longer.
    """

    if ratio_min <= 0:
        raise ValueError("ratio_min must be > 0.")
    if ratio_max <= ratio_min:
        raise ValueError("ratio_max must be greater than ratio_min.")

    scale = short_scale if short_scale is not None else max(short_target / 2, 1.0)
    rewards = []

    for short_resp, long_resp in zip(short_responses, long_responses):
        short_text = short_resp.rstrip()
        long_text = long_resp.rstrip()
        short_len = len(short_text)
        long_len = len(long_text)

        if short_len == 0 or long_len == 0:
            rewards.append(-1.0)
            continue

        ratio = long_len / max(short_len, 1)
        if ratio_min <= ratio <= ratio_max:
            ratio_score = 1.0
        elif ratio < ratio_min:
            ratio_score = 1.0 - (ratio_min - ratio) / ratio_min
        else:
            ratio_score = 1.0 - (ratio - ratio_max) / ratio_max
        ratio_score = max(-1.0, ratio_score)

        short_score = 1.0 - abs(short_len - short_target) / scale
        short_score = max(-1.0, min(short_score, 1.0))

        combined = 0.5 * (ratio_score + short_score)
        rewards.append(float(max(-1.0, min(combined, 1.0))))

    return rewards


def build_prompt_formatter(tokenizer):
    def _formatter(example):
        prompt = example.get("prompt")
        if prompt is None:
            raise KeyError("Expected 'prompt' field in dataset example.")

        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            messages = [
                {
                    "role": "system",
                    "content": "You summarize Reddit posts into TL;DRs.",
                },
                {"role": "user", "content": prompt},
            ]
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    return _formatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train two agents on TL;DR prompts so the second response is 2-3x longer than the first."
        )
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", type=str, default="./mappo_tldr")
    parser.add_argument("--dataset-size", type=int, default=320)
    parser.add_argument("--num-train-epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--ratio-min", type=float, default=2.0)
    parser.add_argument("--ratio-max", type=float, default=3.0)
    parser.add_argument("--short-target-chars", type=int, default=220)
    parser.add_argument("--short-target-scale", type=float, default=None)
    parser.add_argument("--wandb-project", type=str, default="mappo")
    parser.add_argument("--wandb-entity", type=str, default="OpenMLRL")
    parser.add_argument("--wandb-run-name", type=str, default="mappo_tldr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    if args.ratio_min <= 0:
        raise ValueError("ratio_min must be > 0.")
    if args.ratio_max <= args.ratio_min:
        raise ValueError("ratio_max must be greater than ratio_min.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("trl-lib/tldr", split="train")
    usable = min(args.dataset_size, len(dataset))
    dataset = dataset.select(range(usable))

    config = MAGRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        report_to="none",
    )

    wandb_config = {
        "entity": args.wandb_entity,
        "project": args.wandb_project,
        "name": args.wandb_run_name,
    }

    reward_fn = partial(
        dual_length_reward,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        short_target=args.short_target_chars,
        short_scale=args.short_target_scale,
    )

    trainer = MAGRPOTrainer(
        model=args.model_name,
        num_agents=2,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        formatters=build_prompt_formatter(tokenizer),
        args=config,
        train_dataset=dataset,
        wandb_config=wandb_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
