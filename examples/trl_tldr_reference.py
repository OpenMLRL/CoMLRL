"""Train a language model with TRL's PPOTrainer to target 200-character TL;DRs.

The script mirrors the dataset slice, generation parameters, and reward
definition used in the IPPO example so you can compare behaviours. It is
compatible with both the current TRL API (>=0.9) and older releases that expect
`reward_model`, `value_model`, or `train_dataset` positional arguments.

Usage:
    python examples/trl_tldr_reference.py

Ensure that `trl`, `datasets`, `transformers`, and a CUDA-capable PyTorch are
installed in the active environment.
"""

from __future__ import annotations

import inspect
from functools import partial
from typing import Dict, Iterable, List, Optional
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
except ImportError as exc:  # pragma: no cover - informative error message
    raise ImportError(
        "This example requires the `trl` package. Install it with `pip install trl`."
    ) from exc


def build_dataset(max_rows: int = 20) -> Dataset:
    """Load a small TL;DR slice and expose prompts under the `query` key."""

    dataset = load_dataset("trl-lib/tldr", split="train").select(range(max_rows))
    if "prompt" not in dataset.column_names:
        raise ValueError(
            "Expected a `prompt` column in trl-lib/tldr. Columns present: "
            f"{dataset.column_names}"
        )
    dataset = dataset.rename_column("prompt", "query")
    dataset.set_format(type="python")
    return dataset


def make_generation_kwargs(tokenizer) -> Dict:
    """Match the generation settings used in the IPPO example."""

    return {
        "max_new_tokens": 96,
        "do_sample": False,
        "temperature": 0.3,
        "top_p": 0.6,
        "pad_token_id": tokenizer.pad_token_id,
    }


def length_reward(
    target: int, scale: float, prompts: List[str], responses: List[str]
) -> List[float]:
    """Reward sequences whose character length stays near `target`."""

    rewards: List[float] = []
    for response in responses:
        normalized = response.strip()
        reward = -abs(len(normalized) - target) / scale
        rewards.append(float(reward))
    return rewards


def create_trainer(
    config: PPOConfig,
    model: AutoModelForCausalLMWithValueHead,
    ref_model: Optional[AutoModelForCausalLMWithValueHead],
    tokenizer: AutoTokenizer,
    dataset: Dataset,
):
    """Instantiate PPOTrainer regardless of TRL version quirks."""

    params = inspect.signature(PPOTrainer.__init__).parameters
    param_names = list(params.keys())

    def build_kwargs() -> Dict:
        kw: Dict = {}
        if "ref_model" in params:
            kw["ref_model"] = ref_model
        elif "model_ref" in params:
            kw["model_ref"] = ref_model

        if "tokenizer" in params:
            kw["tokenizer"] = tokenizer
        if "dataset" in params:
            kw["dataset"] = dataset
        if "train_dataset" in params:
            kw["train_dataset"] = dataset
        if "reward_model" in params:
            kw.setdefault("reward_model", None)
        if "value_model" in params:
            kw.setdefault("value_model", None)
        return kw

    attempts = []

    kw = build_kwargs()
    if "config" in params:
        kw_config = dict(kw)
        kw_config["config"] = config
        if "model" in params:
            kw_config["model"] = model
            attempts.append(((), kw_config))
        else:
            attempts.append(((model,), kw_config))

    if "ppo_config" in params:
        kw_ppo = dict(build_kwargs())
        kw_ppo["ppo_config"] = config
        if "model" in params:
            kw_ppo["model"] = model
            attempts.append(((), kw_ppo))
        else:
            attempts.append(((model,), kw_ppo))

    base_kw = build_kwargs()
    if "model" in params:
        base_kw["model"] = model
        attempts.append(((config,), base_kw))
        attempts.append(((model,), base_kw))
    else:
        attempts.append(((config, model), base_kw))
        attempts.append(((model,), base_kw))

    last_error: Optional[Exception] = None
    for args, kwargs in attempts:
        try:
            return PPOTrainer(*args, **kwargs)
        except TypeError as err:
            last_error = err
            continue

    raise RuntimeError(
        "Unable to instantiate PPOTrainer. Encountered parameter names: "
        f"{param_names}. Last error: {last_error}"
    )


def iterate_batches(
    dataset: Dataset, batch_size: int
) -> Iterable[Dict[str, List[str]]]:
    """Yield dataset chunks matching PPOTrainer expectations."""

    batch: Dict[str, List[str]] = {"query": []}
    for record in dataset:
        batch["query"].append(record["query"])
        if len(batch["query"]) == batch_size:
            yield batch
            batch = {"query": []}
    if batch["query"]:
        yield batch


def main() -> None:
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = PPOConfig()
    config_kwargs = {
        "model_name": model_name,
        "batch_size": 1,
        "mini_batch_size": 1,
        "ppo_epochs": 4,
        "learning_rate": 1e-6,
        "seed": 42,
        "log_with": "wandb",
        "project_kwargs": {"project": "mlrl", "name": "trl_length_ppo"},
    }
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    dataset = build_dataset()
    generation_kwargs = make_generation_kwargs(tokenizer)
    reward_fn = partial(length_reward, 200, 50.0)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    ppo_trainer = create_trainer(config, model, ref_model, tokenizer, dataset)

    device = ppo_trainer.accelerator.device

    for batch in iterate_batches(dataset, config.batch_size):
        prompt_tensors = tokenizer(
            batch["query"], padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to(device)

        if hasattr(ppo_trainer, "generate"):
            response_tensors = ppo_trainer.generate(prompt_tensors, **generation_kwargs)
        else:
            response_tensors = ppo_trainer.model.generate(
                prompt_tensors, **generation_kwargs
            )

        responses = tokenizer.batch_decode(
            response_tensors[:, prompt_tensors.shape[1] :], skip_special_tokens=True
        )
        rewards = reward_fn(batch["query"], responses)

        stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
        log_batch = {"query": batch["query"], "response": responses}
        ppo_trainer.log_stats(stats, log_batch, rewards)


if __name__ == "__main__":
    main()
