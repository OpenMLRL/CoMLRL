from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from comlrl.trainers.preference import (
    MADPOConfig,
    MADPOTrainer,
    MADPOIterConfig,
    MADPOIterTrainer,
    MARLHFConfig,
    MARLHFTrainer,
    MARLHFIterConfig,
    MARLHFIterTrainer,
    TaggedCentralizedComparatorAdapter,
)
from comlrl.trainers.preference.collaboration import CentralizedCollaboration
from comlrl.trainers.preference.marlhf import JointRewardModel
from comlrl.utils.reward_utils import set_reward_range

GOOD = "<agent_0> good </agent_0> <agent_1> good </agent_1>"
BAD = "<agent_0> bad </agent_0> <agent_1> bad </agent_1>"


class Adapter(TaggedCentralizedComparatorAdapter):
    def build_prompt(self, item, agent_prompts):
        assert agent_prompts == ["left", "right"]
        return "joint left right"


@pytest.fixture
def tokenizer():
    tokens = [
        "<pad>",
        "<eos>",
        "<unk>",
        "joint",
        "left",
        "right",
        "good",
        "bad",
        "<agent_0>",
        "</agent_0>",
        "<agent_1>",
        "</agent_1>",
    ]
    backend = Tokenizer(
        WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def model_for(tokenizer):
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=256,
            n_embd=16,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )


def task_reward(left, right, *, batch_items, prompts):
    assert prompts == ["left"]
    assert batch_items[0]["id"] == "task"
    return [float(a == "good") + float(b == "good") for a, b in zip(left, right)]


set_reward_range(task_reward, 0, 2)


def make_trainer(cls, config_cls, tokenizer, tmp_path, **overrides):
    config = dict(
        collaboration_mode="centralized",
        agent_devices="cpu",
        num_agents=2,
        num_train_epochs=1,
        preference_num_candidates=2,
        preference_pairs_per_sample=1,
        eval_interval=0,
        rollout_buffer_size=1,
        train_batch_size=1,
        max_new_tokens=8,
        agent_learning_rate=1e-3,
    )
    if issubclass(config_cls, MARLHFConfig):
        config.update(
            reward_model_device="cpu", reward_num_train_epochs=1, num_generations=2
        )
    if issubclass(config_cls, (MADPOIterConfig, MARLHFIterConfig)):
        config.update(
            num_iterations=2,
            preference_replay_dir=str(tmp_path / "replay"),
            policy_checkpoint_dir=str(tmp_path / "checkpoints"),
            log_reward_distribution=False,
        )
    config.update(overrides)
    dataset = Dataset.from_list([{"id": "task", "prompt": "task prompt"}])
    model = model_for(tokenizer)
    calls = []

    def generate(*, input_ids, num_return_sequences, **kwargs):
        calls.append(input_ids.clone())
        sequences = []
        for idx in range(num_return_sequences):
            # Target/comparator calls get different pairs without any network/model download.
            text = GOOD if (idx + len(calls)) % 2 else BAD
            ids = torch.tensor(tokenizer.encode(text), device=input_ids.device)
            sequences.append(torch.cat([input_ids[0], ids]))
        return SimpleNamespace(sequences=torch.stack(sequences))

    model.generate = generate
    trainer = cls(
        agents=[model],
        num_agents=2,
        tokenizer=tokenizer,
        args=config_cls(**config),
        train_dataset=dataset,
        eval_dataset=dataset,
        reward_func=task_reward,
        formatters=[lambda item: "left", lambda item: "right"],
        centralized_comparator_adapter=Adapter(),
    )
    trainer.verbose = False
    return trainer, calls


@pytest.mark.parametrize(
    "config_cls", [MADPOConfig, MARLHFConfig, MADPOIterConfig, MARLHFIterConfig]
)
def test_config_defaults_and_centralized_comparator(config_cls):
    assert config_cls(agent_devices="cpu").collaboration_mode == "decentralized"
    config = config_cls(agent_devices="cpu", collaboration_mode="centralized")
    assert config.num_agents == 2
    if hasattr(config, "comparator_generation_mode"):
        assert config.comparator_generation_mode == "centralized"
    with pytest.raises(ValueError, match="collaboration_mode"):
        config_cls(collaboration_mode="misspelled")


def test_single_actor_constraint(tokenizer, tmp_path):
    with pytest.raises(ValueError, match="exactly one actor"):
        MADPOTrainer(
            agents=[model_for(tokenizer), model_for(tokenizer)],
            args=MADPOConfig(collaboration_mode="centralized", agent_devices="cpu"),
        )
    with pytest.raises(ValueError, match="actor index 0"):
        MADPOIterConfig(
            collaboration_mode="centralized", comparator_centralized_agent_index=1
        )


def test_centralized_reference_uses_one_device(tokenizer, tmp_path):
    trainer, _ = make_trainer(
        MADPOTrainer,
        MADPOConfig,
        tokenizer,
        tmp_path,
        reference_kl_enabled=True,
        reference_devices=["cpu"],
    )
    assert len(trainer.reference_models) == 1


@pytest.mark.parametrize(
    "cls,config_cls", [(MADPOTrainer, MADPOConfig), (MADPOIterTrainer, MADPOIterConfig)]
)
def test_madpo_trains_one_joint_actor(cls, config_cls, tokenizer, tmp_path):
    trainer, calls = make_trainer(cls, config_cls, tokenizer, tmp_path)
    assert len(trainer.agents) == len(trainer.optimizers) == trainer.num_agents == 1
    before = trainer.agents[0].transformer.wte.weight.detach().clone()
    trainer.train()
    assert calls and tokenizer.decode(calls[0][0]) == "joint left right"
    assert not torch.equal(before, trainer.agents[0].transformer.wte.weight)
    assert trainer.env_step == (4 if cls is MADPOIterTrainer else 2)
    assert trainer.reward_func.reward_range == (0.0, 2.0)


def test_joint_logprob_covers_both_roles_and_last_token(tokenizer, tmp_path):
    trainer, _ = make_trainer(MADPOTrainer, MADPOConfig, tokenizer, tmp_path)
    prompt = torch.tensor(tokenizer.encode("joint left right"))
    completion = torch.tensor(tokenizer.encode(GOOD))
    full = torch.cat([prompt, completion]).unsqueeze(0)
    model = trainer.agents[0]
    model.train()
    logits = model(full, use_cache=False).logits[0, len(prompt) - 1 : -1]
    expected = logits.log_softmax(-1).gather(1, completion[:, None]).sum()
    actual = trainer._sequence_log_prob(0, prompt, completion)
    torch.testing.assert_close(actual, expected)
    (-actual).backward()
    for text in ["<agent_0>", "<agent_1>", "</agent_1>"]:
        assert (
            model.transformer.wte.weight.grad[tokenizer.convert_tokens_to_ids(text)]
            .abs()
            .sum()
            > 0
        )


def test_marlhf_joint_loss_uses_actor_device(tokenizer, tmp_path):
    trainer, _ = make_trainer(MARLHFTrainer, MARLHFConfig, tokenizer, tmp_path)
    trainer.device = torch.device("meta")
    data = {
        "prompt_input_ids": torch.tensor([tokenizer.encode("joint left right")]),
        "completion_input_ids": [
            [torch.tensor(tokenizer.encode(GOOD)), torch.tensor(tokenizer.encode(BAD))]
        ],
    }
    loss = trainer._compute_loss_with_gradients(trainer.agents[0], data, [1.0, 0.0])
    assert loss.device.type == "cpu"
    loss.backward()


def test_replay_stores_raw_joint_text_and_current_comparator_uses_same_prompt(
    tokenizer, tmp_path
):
    trainer, calls = make_trainer(
        MADPOIterTrainer, MADPOIterConfig, tokenizer, tmp_path
    )
    pairs = trainer._build_preference_dataset(iteration_idx=0)
    assert pairs[0].winner_completions == [GOOD]
    assert pairs[0].loser_completions == [BAD]
    assert len(pairs[0].agent_tensors) == 1
    torch.testing.assert_close(calls[0], calls[1])
    shard = trainer._write_iteration_preference_pairs(0, pairs)
    loaded = trainer._load_replay_shard(shard)
    assert loaded[0].winner_completions == [GOOD]
    torch.testing.assert_close(
        loaded[0].agent_tensors[0].winner_completion_ids,
        pairs[0].agent_tensors[0].winner_completion_ids,
    )
    path = trainer._save_iteration_policy_checkpoint(0)
    assert (Path(path) / "agent_0").is_dir()
    assert not (Path(path) / "agent_1").exists()
    restored = trainer._load_single_policy_checkpoint_agent(path, 0)
    torch.testing.assert_close(
        restored.transformer.wte.weight, trainer.agents[0].transformer.wte.weight
    )


@pytest.mark.parametrize(
    "cls,config_cls",
    [
        (MADPOTrainer, MADPOConfig),
        (MARLHFTrainer, MARLHFConfig),
        (MADPOIterTrainer, MADPOIterConfig),
        (MARLHFIterTrainer, MARLHFIterConfig),
    ],
)
def test_eval_splits_roles_and_uses_task_reward(cls, config_cls, tokenizer, tmp_path):
    trainer, _ = make_trainer(cls, config_cls, tokenizer, tmp_path)
    observed = []
    trainer.eval_logger = lambda **kwargs: observed.append(kwargs) or {"metric": 1.0}
    trainer.eval_aggregator = lambda result, **kwargs: {
        "turn_1/domain_metric": result["metric"]
    }
    if isinstance(trainer, MARLHFTrainer):
        trainer._reward_model_active = True
    result = trainer.evaluate(num_eval_samples=1)
    assert observed[0]["agent_completions_turns"] == [[["good"]], [["good"]]]
    assert result["eval/turn_1/reward_mean"] == 2.0
    assert result["eval/turn_1/domain_metric"] == 1.0
    assert not trainer._centralized_eval_items


@pytest.mark.parametrize(
    "cls,config_cls",
    [(MARLHFTrainer, MARLHFConfig), (MARLHFIterTrainer, MARLHFIterConfig)],
)
def test_marlhf_reward_model_and_policy_train_on_joint_text(
    cls, config_cls, tokenizer, tmp_path
):
    trainer, calls = make_trainer(cls, config_cls, tokenizer, tmp_path)
    reward_models = []

    def init_reward_model():
        trainer.reward_model = JointRewardModel(model_for(tokenizer))
        trainer.reward_tokenizer = tokenizer
        trainer.reward_optimizer = torch.optim.AdamW(
            trainer.reward_model.parameters(), lr=1e-3
        )
        reward_models.append(
            (
                trainer.reward_model,
                trainer.reward_model.reward_head.weight.detach().clone(),
            )
        )

    trainer._init_reward_model = init_reward_model
    before = trainer.agents[0].transformer.wte.weight.detach().clone()
    trainer.train()
    assert len(reward_models) == (2 if cls is MARLHFIterTrainer else 1)
    assert trainer.env_step == (4 if cls is MARLHFIterTrainer else 2)
    assert not torch.equal(before, trainer.agents[0].transformer.wte.weight)
    assert all(
        not torch.equal(model.reward_head.weight, initial)
        for model, initial in reward_models
    )
    assert len(trainer.optimizers) == 1


def test_parse_failures_and_ac_metrics_stay_at_environment_boundary():
    collaboration = CentralizedCollaboration(
        Adapter(), [lambda _: "left", lambda _: "right"], task_reward, 2
    )
    item = {"id": "task"}
    assert collaboration(["invalid"], batch_items=[item]) == [0.0]
    original = SimpleNamespace(
        agent_idx=0, completion=GOOD, metadata={"batch_item": item, "generation_idx": 1}
    )
    result = collaboration.wrap_metrics_callback(lambda samples: samples)([original])
    assert [(sample.agent_idx, sample.completion) for sample in result] == [
        (0, "good"),
        (1, "good"),
    ]
    assert original.completion == GOOD
    assert all(sample.metadata is original.metadata for sample in result)


@pytest.mark.parametrize(
    "mode,options",
    [
        ("nearest_k", {"preference_replay_k": 1}),
        ("all_history", {}),
        ("lambda_decay", {"preference_replay_lambda": 0.8}),
    ],
)
def test_centralized_replay_modes_train_one_actor(mode, options, tokenizer, tmp_path):
    trainer, _ = make_trainer(
        MADPOIterTrainer,
        MADPOIterConfig,
        tokenizer,
        tmp_path,
        preference_replay_mode=mode,
        **options,
    )
    trainer.train()
    assert trainer.env_step == 4
    assert len(trainer.agents) == len(trainer.optimizers) == 1


@pytest.mark.parametrize(
    "policy", ["current", "current_copy", "history", "model", "api"]
)
def test_comparator_sources_keep_joint_text(policy, tokenizer, tmp_path, monkeypatch):
    options = {"comparator_policy": policy}
    if policy == "model":
        model_for(tokenizer).save_pretrained(tmp_path / "external")
        options["comparator_model_name"] = str(tmp_path / "external")
    if policy == "api":
        options["comparator_api_url"] = "https://example.invalid/completions"
    trainer, _ = make_trainer(
        MADPOIterTrainer, MADPOIterConfig, tokenizer, tmp_path, **options
    )
    if policy == "history":
        trainer._save_initial_policy_checkpoint()

    seen_prompts = []

    def generate(agent, items, **kwargs):
        seen_prompts.append(kwargs["prompts_override"][0])
        assert kwargs["agent_idx"] == 0
        return {"completions": [[GOOD, BAD]]}

    def api(**kwargs):
        seen_prompts.append(kwargs["prompt"])
        assert kwargs["agent_idx"] == 0
        return [GOOD, BAD]

    monkeypatch.setattr(trainer, "_generate_completions", generate)
    monkeypatch.setattr(trainer, "_call_comparator_api", api)
    outputs = trainer._generate_comparator_outputs_for_item(
        {"id": "task"},
        iteration_idx=0,
        num_candidates=2,
    )
    assert seen_prompts == ["joint left right"]
    assert len(outputs) == 1
    assert outputs[0]["completions"] == [[GOOD, BAD]]
    assert trainer._compute_raw_and_processed_rewards(
        outputs[0]["prompts"],
        [[GOOD, BAD]],
        batch_items=[{"id": "task"}],
    ) == ([2.0, 0.0], [2.0, 0.0])
