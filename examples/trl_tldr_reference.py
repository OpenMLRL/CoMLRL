"""
Reference TL;DR PPO fine-tuning using TRL's PPOTrainer.

This script mirrors the dataset slice, generation settings, and reward used in
our IPPO example so that you can compare behaviours. It follows the current
TRL docs (https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo_tldr.py).
If your installed TRL version exposes an older PPOTrainer signature, the script
will emit a clear error asking you to upgrade.
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import Dict, List
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
except ImportError as exc:  # pragma: no cover - informative error message
    raise ImportError(
        "This example requires the `trl` package. Install with `pip install trl`."
    ) from exc


def tldr_reward(prompts: List[str], responses: List[str]) -> List[float]:
    rewards: List[float] = []
    for response in responses:
        normalized = response.strip()
        generation_length = len(normalized)
        reward = -abs(generation_length - 200) / 50.0
        rewards.append(float(reward))
    return rewards


def build_dataset(slice_size: int = 20) -> Dataset:
    dataset = load_dataset("trl-lib/tldr", split="train").select(range(slice_size))
    dataset = dataset.rename_column("prompt", "query")
    return dataset


def prepare_generation_kwargs(tokenizer) -> Dict:
    return {
        "max_new_tokens": 96,
        "do_sample": False,
        "temperature": 0.3,
        "top_p": 0.6,
        "pad_token_id": tokenizer.pad_token_id,
    }


def main() -> None:
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create config aligned with the documentation.
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-6,
        batch_size=1,
        mini_batch_size=1,
        ppo_epochs=4,
        log_with="wandb",
        accelerator_kwargs={"mixed_precision": "bf16"},
        project_kwargs={"project": "mlrl", "name": "trl_ippo_reference"},
        seed=42,
    )

    # Build dataset slice identical to the IPPO example.
    dataset = build_dataset()

    # Prepare value-head model and (optional) reference model.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Detect PPOTrainer signature at runtime for compatibility.
    ppo_init_params = inspect.signature(PPOTrainer.__init__).parameters
    if {"reward_model", "value_model", "train_dataset"} <= ppo_init_params.keys():
        raise RuntimeError(
            "Your installed `trl` version exposes the legacy PPOTrainer signature "
            "that requires reward/value models. Please upgrade to `trl>=0.9` (or "
            "a version that matches the TL;DR example) to run this script."
        )

    generation_kwargs = prepare_generation_kwargs(tokenizer)

    # Use the documented constructor (config, model, tokenizer, dataset…)
    try:
        ppo_trainer = PPOTrainer(
            ppo_config,
            model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
        )
    except TypeError as exc:
        raise RuntimeError(
            "PPOTrainer initialisation failed. Please ensure your `trl` version "
            "matches the documentation example (>=0.9)."
        ) from exc

    reward_fn = partial(tldr_reward)

    for batch in ppo_trainer.dataloader:
        prompts = batch["query"]
        query_tensors = tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to(ppo_trainer.accelerator.device)

        response_tensors = ppo_trainer.model.generate(
            query_tensors, **generation_kwargs
        )

        responses = tokenizer.batch_decode(
            response_tensors[:, query_tensors.shape[1] :],
            skip_special_tokens=True,
        )
        rewards = reward_fn(prompts, responses)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        log_batch = {"query": prompts, "response": responses}
        ppo_trainer.log_stats(stats, log_batch, rewards)


if __name__ == "__main__":
    main()
