"""Real GPT-2 training resumes exactly; COMLRL_TEST_DEVICE defaults to cpu."""

import copy
import os

from datasets import Dataset
import pytest
import torch

from comlrl.trainers.actor_critic import IACConfig, IACTrainer, MAACConfig, MAACTrainer
from comlrl.trainers.reinforce import (
    MAGRPOConfig,
    MAGRPOTrainer,
    MAREINFORCEConfig,
    MAREINFORCETrainer,
)
from comlrl.utils.checkpoint import _decode
from comlrl.utils.reward_utils import set_reward_range

from checkpoint_helpers import Interrupted
from helpers import (
    assert_nested_equal,
    model_for,
    restore_rng_state,
    rng_state,
    seed_all,
)


class SplitRoles:
    """A local environment accepts any sampled text as a joint response."""

    def build_prompt(self, item, agent_prompts):
        return "joint " + " ".join(agent_prompts)

    def parse_completion(self, completion, item, num_agents):
        words = completion.split()
        return [" ".join(words[index::num_agents]) for index in range(num_agents)]


def text_reward(*agent_completions, **kwargs):
    # Distinguish sampled responses without a remote judge or model dependency.
    return [
        1.0 + sum(sum(map(ord, text)) for text in responses) / 1000.0
        for responses in zip(*agent_completions)
    ]


set_reward_range(text_reward, 0, 100)


def format_prompt(item):
    return item["prompt"]


@pytest.fixture
def test_device():
    device = torch.device(os.environ.get("COMLRL_TEST_DEVICE", "cpu"))
    if device.type == "cpu":
        yield device
        return
    if device.type != "cuda":
        pytest.fail("COMLRL_TEST_DEVICE must be cpu or a CUDA device.")
    if not torch.cuda.is_available():
        pytest.fail("COMLRL_TEST_DEVICE requests CUDA, but CUDA is unavailable.")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index >= torch.cuda.device_count():
        pytest.fail(f"COMLRL_TEST_DEVICE requests unavailable CUDA device {index}.")
    device = torch.device("cuda", index)
    torch.cuda.init()
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    # Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the CUDA test process.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield device
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cuda.matmul.allow_tf32 = previous[2]
        torch.backends.cudnn.allow_tf32 = previous[3]
        torch.backends.cudnn.benchmark = previous[4]
        torch.backends.cudnn.deterministic = previous[5]


@pytest.fixture(autouse=True)
def preserve_rng(test_device):
    previous = rng_state()
    try:
        yield
    finally:
        restore_rng_state(previous)


def make_trainer(variant, tokenizer, seed, device):
    seed_all(seed)
    tokenizer = copy.deepcopy(tokenizer)
    dataset = Dataset.from_dict({"prompt": ["left", "right"]})
    common = dict(
        num_agents=2,
        num_turns=1,
        num_train_epochs=2,
        num_generations=2,
        # MAGRPO scores all but its trailing token, so retain multiple tokens.
        max_new_tokens=3,
        temperature=0.9,
        top_p=1.0,
        top_k=0,
        agent_devices=str(device),
        agent_learning_rate=1e-3,
        eval_interval=0,
        rollout_buffer_size=3,
        train_batch_size=2,
        early_termination_threshold=None,
        reference_kl_enabled=True,
        reference_kl_coef=0.05,
        reference_devices=str(device),
    )
    kwargs = dict(
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_func=text_reward,
        formatters=format_prompt,
    )
    if variant == "centralized-magrpo":
        trainer = MAGRPOTrainer(
            agents=[model_for(tokenizer)],
            num_agents=2,
            args=MAGRPOConfig(
                **{**common, "collaboration_mode": "centralized", "num_turns": 1}
            ),
            centralized_adapter=SplitRoles(),
            **kwargs,
        )
    else:
        kwargs["agents"] = [model_for(tokenizer) for _ in range(2)]
        if variant in {"magrpo", "mareinforce"}:
            trainer_cls, config_cls = (
                (MAGRPOTrainer, MAGRPOConfig)
                if variant == "magrpo"
                else (MAREINFORCETrainer, MAREINFORCEConfig)
            )
            trainer = trainer_cls(num_agents=2, args=config_cls(**common), **kwargs)
        elif variant.startswith("iac"):
            separate = variant == "iac-separate"
            if separate:
                kwargs["critics"] = [model_for(tokenizer) for _ in range(2)]
            trainer = IACTrainer(
                args=IACConfig(
                    use_separate_critic=separate,
                    critic_devices=str(device),
                    critic_learning_rate=1e-3,
                    **common,
                ),
                **kwargs,
            )
        else:
            assert variant == "maac"
            trainer = MAACTrainer(
                critics=[model_for(tokenizer)],
                args=MAACConfig(
                    critic_devices=str(device), critic_learning_rate=1e-3, **common
                ),
                **kwargs,
            )
    trainer.verbose = False
    for name in ("agents", "critics", "reference_models"):
        for model in getattr(trainer, name, []):
            assert all(parameter.device == device for parameter in model.parameters())
    return trainer


def model_states(trainer):
    return {
        name: [
            copy.deepcopy(model.state_dict()) for model in getattr(trainer, name, [])
        ]
        for name in ("agents", "critics", "reference_models")
    }


def optimizer_states(trainer):
    states = {}
    for name in (
        "optimizers",
        "agent_optimizers",
        "critic_optimizers",
        "critic_optimizer",
    ):
        value = getattr(trainer, name, None)
        if value is not None:
            optimizers = value if isinstance(value, list) else [value]
            states[name] = [optimizer.state_dict() for optimizer in optimizers]
    return states


@pytest.mark.parametrize(
    "variant",
    [
        "magrpo",
        "mareinforce",
        "centralized-magrpo",
        "iac-shared",
        "iac-separate",
        "maac",
    ],
)
def test_real_model_resume_matches_two_uninterrupted_epochs(
    variant, tokenizer, tmp_path, test_device
):
    baseline = make_trainer(variant, tokenizer, seed=37, device=test_device)
    initial = model_states(baseline)
    seed_all(83)
    baseline.train()
    expected_rng = rng_state()
    expected = model_states(baseline)
    expected_optimizers = optimizer_states(baseline)

    # Confirm that this is a learning test, rather than two identical no-op runs.
    for before, after in zip(initial["agents"], expected["agents"]):
        assert any(
            not torch.equal(after[name], tensor) for name, tensor in before.items()
        )
    if variant == "iac-shared":
        value_before, value_after = initial["agents"], expected["agents"]
    else:
        value_before, value_after = initial["critics"], expected["critics"]
    for before, after in zip(value_before, value_after):
        assert any(
            "value_head" in name and not torch.equal(after[name], tensor)
            for name, tensor in before.items()
        )
    assert_nested_equal(expected["reference_models"], initial["reference_models"])
    assert all(
        any(
            torch.count_nonzero(state["exp_avg"]).item() > 0
            for state in optimizer["state"].values()
        )
        for optimizers in expected_optimizers.values()
        for optimizer in optimizers
    )

    interrupted = make_trainer(variant, tokenizer, seed=37, device=test_device)
    commit = interrupted._checkpoint_commit

    def stop_after_first_epoch(progress, force=False):
        commit(progress, force=force)
        if progress["epoch"] == 1:
            raise Interrupted()

    interrupted._checkpoint_commit = stop_after_first_epoch
    seed_all(83)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path / "interrupted")
    checkpoint = tmp_path / "interrupted" / "checkpoint-1"
    assert (checkpoint / "training_state.pt").is_file()
    if test_device.type == "cuda":
        saved = _decode(
            torch.load(
                checkpoint / "training_state.pt", map_location="cpu", weights_only=True
            )
        )
        assert len(saved["rng"]["cuda"]) == torch.cuda.device_count()
        assert_nested_equal(saved["rng"]["cuda"], rng_state()["cuda"])

    resumed = make_trainer(variant, tokenizer, seed=991, device=test_device)
    assert any(
        not torch.equal(first, second)
        for first, second in zip(
            resumed.agents[0].parameters(), baseline.agents[0].parameters()
        )
    )
    seed_all(1021)
    resumed.train(resume_from_checkpoint=checkpoint)

    assert_nested_equal(model_states(resumed), expected)
    assert_nested_equal(optimizer_states(resumed), expected_optimizers)
    assert resumed.env_step == baseline.env_step > interrupted.env_step > 0
    assert resumed._last_train_log_step == baseline._last_train_log_step
    assert (
        resumed.rollout_buffers
        == baseline.rollout_buffers
        == [[]] * len(resumed.agents)
    )
    assert resumed._checkpoint_progress["epoch"] == 2
    assert_nested_equal(rng_state(), expected_rng)
