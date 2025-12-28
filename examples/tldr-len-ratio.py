import argparse
from functools import partial
from typing import List

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


def rollout_metrics(rollouts):
    if not rollouts:
        return {}
    char_lengths = [sample.metadata.get("char_length", 0.0) for sample in rollouts]
    return {"response_char_length_mean": float(sum(char_lengths) / len(char_lengths))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train two IAC agents on TL;DR prompts so the second response is 2-3x longer."
        )
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--critic-model", type=str, default=None)
    parser.add_argument("--separate-critic", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./iac_tldr")
    parser.add_argument("--dataset-size", type=int, default=320)
    parser.add_argument("--num-train-epochs", type=int, default=15)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-6)
    parser.add_argument("--critic-learning-rate", type=float, default=1e-6)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--rollout-buffer-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--ratio-min", type=float, default=2.0)
    parser.add_argument("--ratio-max", type=float, default=3.0)
    parser.add_argument("--short-target-chars", type=int, default=220)
    parser.add_argument("--short-target-scale", type=float, default=None)
    parser.add_argument("--wandb-project", type=str, default="iac")
    parser.add_argument("--wandb-entity", type=str, default="OpenMLRL")
    parser.add_argument("--wandb-run-name", type=str, default="iac_tldr")
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

    config = IACConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        value_loss_coef=args.value_loss_coef,
        rollout_buffer_size=args.rollout_buffer_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_separate_critic=args.separate_critic,
        critic_model_name_or_path=args.critic_model,
        num_agents=2,
        num_turns=1,
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

    trainer = IACTrainer(
        model=args.model_name,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        formatters=build_prompt_formatters(tokenizer),
        args=config,
        train_dataset=dataset,
        wandb_config=wandb_config,
        metrics_callback=rollout_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
