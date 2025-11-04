"""
Minimal TL;DR summarization example using the IPPOTrainer.

This mirrors the PPO quick-start from TRL but swaps in Qwen-3B and the
parameter-sharing IPPO trainer.
"""

from datasets import Dataset
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
    return example["prompt"]


def main():
    model_name = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "Summarize the following Reddit post in a short TL;DR:\n\n"
        "Title: I finally learned to enjoy cooking\n"
        "Body: For years I relied on takeout, but over the last three months I started "
        "experimenting with meal kits, then farmer's market produce. The first dinners were rough, "
        "yet now my friends actually request seconds. How do I stay motivated through busy work weeks?",
        "Summarize the following Reddit post in a short TL;DR:\n\n"
        "Title: Moving abroad with two cats\n"
        "Body: I'm relocating from Canada to Spain for a two-year contract. I have two indoor cats "
        "that have never flown before. What steps should I take to make the trip humane and stick to EU regulations?",
        "Summarize the following Reddit post in a short TL;DR:\n\n"
        "Title: Startup burnout is real\n"
        "Body: I co-founded a small AI startup last year. The work is meaningful but the 80-hour weeks "
        "and constant fundraising pitches are draining. How do other founders avoid torching both health and relationships?",
        "Summarize the following Reddit post in a short TL;DR:\n\n"
        "Title: Planning a three-day backpacking loop in Yosemite\n"
        "Body: Visiting in late July with two friends. We'd like a route with good alpine lakes, moderate mileage, "
        "and reliable water sources. Any lesser-known loops worth considering?",
        "Summarize the following Reddit post in a short TL;DR:\n\n"
        "Title: Remote onboarding woes\n"
        "Body: Started a new job where everyone is remote. Laptop arrived late, meetings are calendar chaos, "
        "and I haven't met my manager yet. What are the best moves to ramp up without annoying teammates?",
        "Summarize the following Reddit post in a short TL;DR:\n\n"
        "Title: Starting a balcony garden\n"
        "Body: Small south-facing balcony in an apartment. I'd like to grow herbs and a few veggies with minimal gear. "
        "How do I set up containers and soil without overspending?",
    ]

    dataset = Dataset.from_dict({"prompt": prompts})

    config = IPPOConfig(
        output_dir="./ippo_qwen_tldr",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        rollout_buffer_size=2,
        ppo_epochs=2,
        max_new_tokens=128,
        logging_steps=1,
    )

    trainer = IPPOTrainer(
        model=model_name,
        tokenizer=tokenizer,
        reward_func=tldr_reward,
        formatters=prompt_formatter,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
