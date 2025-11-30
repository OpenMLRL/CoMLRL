import argparse
import random
from functools import partial
from typing import List

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

from comlrl.trainers.iac import IACConfig, IACTrainer


def dual_length_reward(
    short_responses: List[str],
    long_responses: List[str],
    ratio_min: float = 2.0,
    ratio_max: float = 3.0,
    short_target: int = 220,
    short_scale: float | None = None,
) -> list[float]:
    """Reward two agents for matching a target length ratio."""
    if ratio_min <= 0:
        raise ValueError("ratio_min must be > 0.")
    if ratio_max <= ratio_min:
        raise ValueError("ratio_max must exceed ratio_min.")

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


def build_prompt_formatters(tokenizer):
    def make_formatter(system_prompt: str):
        def _formatter(example):
            prompt = example.get("prompt")
            if prompt is None:
                raise KeyError("Expected 'prompt' field in dataset example.")

            apply_template = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply_template):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                return apply_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return f"{system_prompt}\n\n{prompt}"

        return _formatter

    concise = "You summarize Reddit posts into concise TL;DRs (~220 characters)."
    detailed = (
        "You summarize Reddit posts into detailed TL;DRs about 2-3x longer than a"
        " standard version."
    )
    return [make_formatter(concise), make_formatter(detailed)]


def value_variance_metrics(rollouts) -> dict[str, float]:
    """Compute variance of the critic value estimates across collected rollouts."""
    if not rollouts:
        return {}
    values = []
    for sample in rollouts:
        val = sample.old_value
        if torch.is_tensor(val):
            values.append(val.view(-1).float())
        else:
            values.append(torch.tensor([float(val)]))
    stacked = torch.cat(values) if values else torch.tensor([])
    variance = (
        torch.var(stacked, unbiased=False).item() if stacked.numel() > 1 else 0.0
    )
    return {"value_variance": float(variance)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IAC (multi-generation) on TL;DR length ratio with aligned sampling."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", type=str, default="./iac_multigen")
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-6)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--rollout-buffer-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--ac-epochs", type=int, default=1)
    parser.add_argument("--ratio-min", type=float, default=2.0)
    parser.add_argument("--ratio-max", type=float, default=3.0)
    parser.add_argument("--short-target-chars", type=int, default=220)
    parser.add_argument("--short-target-scale", type=float, default=None)
    parser.add_argument("--wandb-project", type=str, default="compare")
    parser.add_argument("--wandb-entity", type=str, default="openmlrl")
    parser.add_argument("--wandb-run-name", type=str, default="iac")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("trl-lib/tldr", split="train")
    usable = min(args.dataset_size, len(dataset))
    dataset = dataset.select(range(usable))

    reward_fn = partial(
        dual_length_reward,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        short_target=args.short_target_chars,
        short_scale=args.short_target_scale,
    )

    formatters = build_prompt_formatters(tokenizer)

    use_sampling = args.num_generations > 1
    top_k = args.top_k if use_sampling else None

    iac_trainer = IACTrainer(
        model=args.model_name,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        formatters=formatters,
        metrics_callback=value_variance_metrics,
        args=IACConfig(
            output_dir=f"{args.output_dir}/iac",
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            value_loss_coef=args.value_loss_coef,
            rollout_buffer_size=args.rollout_buffer_size,
            mini_batch_size=args.mini_batch_size,
            ac_epochs=args.ac_epochs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=top_k,
            do_sample=use_sampling,
            num_train_epochs=args.num_train_epochs,
            num_agents=2,
            num_return_sequences=args.num_generations,
        ),
        train_dataset=dataset,
        wandb_config={
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "config_sections": {
                "dataset": {"name": "trl-lib/tldr", "size": args.dataset_size},
                "trainer": {
                    "num_generations": args.num_generations,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                },
            },
        },
    )
    iac_trainer.train()
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
