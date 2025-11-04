"""
Reference TL;DR training loop using TRL's PPOTrainer for comparison.

This mirrors the reward definition and dataset slice used in the IPPO example.
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


def tldr_reward(prompts, responses):
    rewards = []
    for response in responses:
        normalized = response.strip()
        generation_length = len(normalized)
        reward = -abs(generation_length - 200) / 50.0
        rewards.append(float(reward))
    return rewards


def build_dataset(slice_size: int = 20):
    dataset = load_dataset("trl-lib/tldr", split="train").select(range(slice_size))
    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.rename_column("summary", "response")
    dataset.set_format(type="python")
    return dataset


def main():
    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    ppo_config = PPOConfig(
        learning_rate=1e-6,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        target_kl=0.1,
        ppo_epochs=4,
        seed=42,
        log_with="wandb",
    )
    wandb_kwargs = {"project": "mlrl", "name": "trl_ippo_reference"}

    dataset = build_dataset()

    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=value_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        wandb_kwargs=wandb_kwargs,
    )

    generation_kwargs = {
        "max_new_tokens": 96,
        "do_sample": False,
        "temperature": 0.3,
        "top_p": 0.6,
        "pad_token_id": tokenizer.pad_token_id,
    }

    for batch in ppo_trainer.dataloader:
        query_tensors = tokenizer(
            batch["query"], padding=True, truncation=True, return_tensors="pt"
        ).input_ids.to(ppo_trainer.accelerator.device)

        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

        responses = tokenizer.batch_decode(
            response_tensors[:, query_tensors.shape[1] :],
            skip_special_tokens=True,
        )
        rewards = tldr_reward(batch["query"], responses)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


if __name__ == "__main__":
    main()
