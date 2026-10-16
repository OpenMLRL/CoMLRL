import copy
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from comlrl.trainers.preference import (
    MADPOIterTrainer,
    MADPOTrainer,
    MARLHFIterTrainer,
    MARLHFTrainer,
)
from comlrl.trainers.preference.madpo import AgentPreferenceTensors, PreferencePair
from comlrl.utils.checkpoint import TrainingCheckpointMixin


class Interrupted(RuntimeError):
    pass


class TinyPolicy(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False)

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "weights.pt")


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def pair(value):
    tokens = AgentPreferenceTensors(
        torch.tensor([1]), torch.tensor([2]), torch.tensor([3])
    )
    return PreferencePair(
        [str(value)],
        ["winner"],
        ["loser"],
        [tokens],
        float(value),
        0.0,
        float(value),
    )


def toy_trainer(cls=MADPOTrainer):
    trainer = object.__new__(cls)
    trainer.args = SimpleNamespace(
        num_turns=1,
        num_train_epochs=3,
        train_batch_size=2,
        eval_interval=2,
        eval_num_samples=1,
        use_environment_step=True,
        environment_steps_per_pair=1,
    )
    trainer.agents = [TinyPolicy()]
    trainer.optimizers = [torch.optim.AdamW(trainer.agents[0].parameters(), lr=0.03)]
    trainer.reference_models = [copy.deepcopy(trainer.agents[0])]
    trainer.agent_devices = [torch.device("cpu")]
    trainer.num_agents = 1
    trainer.tokenizers = []
    trainer.rollout_buffers = [[]]
    trainer.env_step = 0
    trainer._last_train_log_step = -1
    trainer.train_dataset = None
    trainer.eval_dataset = None
    trainer.wandb_config = None
    trainer.wandb_initialized = False
    trainer.verbose = False
    trainer.events = []
    trainer.evaluate = lambda **kwargs: trainer.events.append(
        ("eval", trainer.env_step)
    )
    trainer._build_preference_dataset = lambda **kwargs: [pair(i) for i in range(1, 7)]

    def update(batch):
        trainer.events.append(tuple(item.prompts[0] for item in batch))
        target = sum(item.winner_reward for item in batch) + torch.rand(()).item()
        optimizer = trainer.optimizers[0]
        optimizer.zero_grad()
        loss = (trainer.agents[0](torch.ones(1, 1)).sum() - target).square()
        loss.backward()
        optimizer.step()
        return {}

    trainer._update_from_preference_batch = update
    return trainer


def stop_after(trainer, predicate):
    commit = trainer._checkpoint_commit

    def interrupt(progress, force=False):
        commit(progress, force=force)
        if predicate(progress):
            raise Interrupted

    trainer._checkpoint_commit = interrupt


def assert_same_training(left, right):
    assert torch.equal(left.agents[0].weight, right.agents[0].weight)
    assert left.env_step == right.env_step
    a = left.optimizers[0].state_dict()["state"]
    b = right.optimizers[0].state_dict()["state"]
    for index in a:
        for key in a[index]:
            assert torch.equal(a[index][key], b[index][key])


def test_madpo_resume_preserves_pairs_shuffle_optimizer_and_rng(tmp_path):
    seed(12)
    baseline = toy_trainer()
    baseline.train()
    expected_random = (random.random(), torch.rand(()).item())

    seed(12)
    interrupted = toy_trainer()
    stop_after(interrupted, lambda progress: progress.get("epoch") == 1)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)

    seed(999)
    resumed = toy_trainer()
    resumed._build_preference_dataset = lambda **kwargs: pytest.fail(
        "regenerated pairs"
    )
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoint-2")
    assert_same_training(baseline, resumed)
    assert baseline.events == interrupted.events + resumed.events
    assert expected_random == (random.random(), torch.rand(()).item())
    assert torch.equal(
        baseline.reference_models[0].weight, resumed.reference_models[0].weight
    )


def iterative_trainer(directory, cls=MADPOIterTrainer, scoring="reward_model"):
    trainer = marlhf_trainer(cls) if cls is MARLHFIterTrainer else toy_trainer(cls)
    trainer.args.num_iterations = 4
    trainer.args.preference_replay_mode = "all_history"
    trainer.args.preference_replay_sample_size = 4
    trainer.args.preference_replay_dir = str(directory / "replay")
    trainer.args.policy_checkpoint_dir = str(directory / "policies")
    trainer.args.comparator_policy = "history"
    trainer.args.comparator_history_k = 1
    trainer.args.preference_scoring_reward = scoring
    trainer._initialize_comparator_rng()
    trainer._reset_iteration_reward_distribution = lambda: None
    trainer._prepare_iteration_current_copy_comparator = lambda: None
    trainer._clear_iteration_current_copy_comparator = lambda: None
    trainer._log_iteration_replay = lambda *args, **kwargs: None
    trainer._preference_tensors_from_text = lambda *args, **kwargs: pair(
        1
    ).agent_tensors[0]

    def build(iteration_idx, **kwargs):
        history = Path(trainer._history_policy_checkpoint_path(iteration_idx))
        assert (history / "agent_0" / "weights.pt").is_file()
        with trainer._comparator_rng(trainer.agents[0]):
            comparator_value = torch.rand(()).item()
        if iteration_idx == 0:
            return []
        reward = getattr(trainer, "reward_model", None)
        previous_reward = 0.0 if reward is None else reward.weight.item()
        return [
            pair(iteration_idx + comparator_value + previous_reward + i)
            for i in range(3)
        ]

    trainer._build_preference_dataset = build
    return trainer


def test_iterative_resume_moves_history_replay_and_comparator_rng(tmp_path):
    seed(25)
    baseline = iterative_trainer(tmp_path / "baseline")
    baseline.train()
    expected_random = (random.random(), torch.rand(()).item())

    seed(25)
    interrupted = iterative_trainer(tmp_path / "original")
    stop_after(interrupted, lambda progress: progress.get("iteration") == 2)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path / "checkpoints")
    assert (tmp_path / "checkpoints" / "checkpoint-1").is_dir()  # Empty iteration.
    portable = tmp_path / "portable"
    shutil.copytree(tmp_path / "checkpoints" / "checkpoint-2", portable)
    shutil.rmtree(tmp_path / "original")
    shutil.rmtree(tmp_path / "checkpoints")

    seed(777)
    resumed = iterative_trainer(tmp_path / "new-run")
    resumed.train(resume_from_checkpoint=portable)
    assert_same_training(baseline, resumed)
    assert baseline.events == interrupted.events + resumed.events
    assert expected_random == (random.random(), torch.rand(()).item())
    assert [shard.iteration for shard in resumed._preference_replay_shards_state] == [
        2,
        3,
        4,
    ]
    assert torch.equal(
        baseline._comparator_rng_states["cpu"],
        resumed._comparator_rng_states[resumed._comparator_rng_streams["0"]["id"]],
    )
    assert not (tmp_path / "original").exists()


def marlhf_trainer(cls=MARLHFTrainer):
    trainer = toy_trainer(cls)
    trainer.args.rl_algorithm = "magrpo"
    trainer.args.advantage_mode = "mean"
    trainer.reward_model = None
    trainer.reward_optimizer = None
    trainer.reward_tokenizer = None
    trainer._reward_model_active = False
    trainer._evaluating_with_task_reward = False

    def initialize():
        trainer.reward_model = TinyPolicy()
        trainer.reward_optimizer = torch.optim.AdamW(
            trainer.reward_model.parameters(), lr=0.02
        )

    def reward_training(pairs):
        trainer.reward_optimizer.zero_grad()
        trainer.reward_model(torch.ones(1, 1)).sum().backward()
        trainer.reward_optimizer.step()
        trainer.reward_model.eval()

    trainer._init_reward_model = initialize
    trainer._train_reward_model = reward_training
    trainer.get_train_dataloader = lambda: [[pair(1)], [pair(2)]]
    trainer._drain_ready_buffers = lambda agents: None

    def online_step(item, **kwargs):
        assert not trainer.reward_model.training
        trainer._update_from_preference_batch([item])
        trainer.env_step += 1

    trainer._train_step_returns = online_step
    return trainer


@pytest.mark.parametrize("boundary", ["reward", "online"])
def test_marlhf_resumes_reward_phase_and_online_epoch(tmp_path, boundary):
    seed(31)
    baseline = marlhf_trainer()
    baseline.train()
    expected_random = (random.random(), torch.rand(()).item())

    seed(31)
    interrupted = marlhf_trainer()
    stop_after(
        interrupted,
        lambda progress: progress.get("phase") == "online_rl"
        and (
            not progress.get("rl")
            if boundary == "reward"
            else progress.get("rl", {}).get("epoch") == 1
        ),
    )
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    checkpoint = tmp_path / ("checkpoint-1" if boundary == "reward" else "checkpoint-2")

    seed(89)
    resumed = marlhf_trainer()
    resumed._build_preference_dataset = lambda **kwargs: pytest.fail(
        "regenerated reward data"
    )
    resumed._train_reward_model = lambda pairs: pytest.fail("retrained reward model")
    resumed.train(resume_from_checkpoint=checkpoint)
    assert_same_training(baseline, resumed)
    assert torch.equal(baseline.reward_model.weight, resumed.reward_model.weight)
    assert baseline.events == interrupted.events + resumed.events
    assert expected_random == (random.random(), torch.rand(()).item())


class TinyActor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, value):
        return self.model(value)


class TinyACTrainer(TrainingCheckpointMixin):
    instances = []

    def __init__(self, *, agents, args, train_dataset, **kwargs):
        self.args = args
        self.train_dataset = train_dataset
        self.agents = [TinyActor(agents[0])]
        self.critics = [TinyPolicy()]
        self.reference_models = [copy.deepcopy(agents[0])]
        self.agent_optimizers = [
            torch.optim.AdamW(self.agents[0].parameters(), lr=0.03)
        ]
        self.critic_optimizer = torch.optim.AdamW(self.critics[0].parameters(), lr=0.02)
        self.env_step = 0
        self.rollout_buffers = [[]]
        self.instances.append(self)

    def train(self):
        self._train_checkpoint_epochs({}, lambda *args, **kwargs: None)

    def _train_checkpoint_epochs(self, progress, commit):
        for epoch in range(progress.get("epoch", 0), self.args.num_train_epochs):
            for _ in range(2):
                value = torch.ones(1, 1)
                target = self.reference_models[0](value).detach() + torch.rand(())
                self.agent_optimizers[0].zero_grad()
                self.critic_optimizer.zero_grad()
                loss = (self.agents[0](value) - target).square().sum()
                loss += (self.critics[0](value) - target).square().sum()
                loss.backward()
                self.agent_optimizers[0].step()
                self.critic_optimizer.step()
                self.env_step += 1
            commit({"kind": "actor_critic", "epoch": epoch + 1}, force=False)


def marlhf_ac_trainer():
    trainer = marlhf_trainer()
    trainer.args.rl_algorithm = "maac"
    trainer._build_actor_critic_common_config = lambda: {
        "num_train_epochs": 3,
        "num_agents": 1,
        "num_turns": 1,
    }
    trainer._build_actor_critic_model_config = lambda: {}
    trainer._resolve_critic_model_source = lambda: "toy"
    trainer.formatters = []
    trainer.metrics_callback = None
    trainer.external_transition = None
    trainer.critic_sources = None
    trainer.centralized_collaboration = None
    return trainer


@pytest.mark.parametrize("boundary", ["reward", "online"])
def test_marlhf_nested_ac_restores_critic_optimizer_references_and_rng(
    tmp_path, monkeypatch, boundary
):
    from comlrl.trainers.preference import marlhf

    monkeypatch.setattr(marlhf, "MAACTrainer", TinyACTrainer)
    seed(42)
    baseline = marlhf_ac_trainer()
    baseline.train()
    baseline_child = TinyACTrainer.instances[-1]
    expected_random = (random.random(), torch.rand(()).item())

    seed(42)
    interrupted = marlhf_ac_trainer()
    stop_after(
        interrupted,
        lambda progress: progress.get("phase") == "online_rl"
        and (
            not progress.get("rl")
            if boundary == "reward"
            else progress.get("rl", {}).get("epoch") == 1
        ),
    )
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path)
    checkpoint = tmp_path / ("checkpoint-1" if boundary == "reward" else "checkpoint-2")

    seed(75)
    resumed = marlhf_ac_trainer()
    resumed.train(resume_from_checkpoint=checkpoint)
    child = resumed._checkpoint_ac_trainer
    assert torch.equal(baseline.agents[0].weight, resumed.agents[0].weight)
    assert torch.equal(baseline_child.critics[0].weight, child.critics[0].weight)
    assert torch.equal(
        baseline_child.reference_models[0].weight, child.reference_models[0].weight
    )
    for name in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(
            baseline_child.critic_optimizer.state_dict()["state"][0][name],
            child.critic_optimizer.state_dict()["state"][0][name],
        )
    assert baseline.env_step == resumed.env_step
    assert expected_random == (random.random(), torch.rand(()).item())


@pytest.mark.parametrize(
    "cls", [MADPOTrainer, MARLHFTrainer, MADPOIterTrainer, MARLHFIterTrainer]
)
@pytest.mark.parametrize("interval", [0, -1, True, False, 1.0, "1", None])
def test_interval_only_is_validated_before_training(cls, interval):
    trainer = object.__new__(cls)
    with pytest.raises(ValueError, match="checkpoint_interval"):
        trainer.train(checkpoint_interval=interval)


@pytest.mark.parametrize("kind", ["madpo", "marlhf", "madpo_iter", "marlhf_iter"])
def test_plain_preference_training_does_not_retain_checkpoint_data(tmp_path, kind):
    if kind == "madpo":
        trainer = toy_trainer()
    elif kind == "marlhf":
        trainer = marlhf_trainer()
    elif kind == "madpo_iter":
        trainer = iterative_trainer(tmp_path / "run")
    else:
        trainer = toy_trainer(MARLHFIterTrainer)
        trainer.args.num_iterations = 0
    trainer.train()
    assert trainer._checkpoint_untracked_preference
    assert not getattr(trainer, "_checkpoint_progress", {}).get("preference_pairs")
    with pytest.raises(RuntimeError, match="checkpoint_dir"):
        trainer.save_checkpoint(tmp_path / "manual")


def test_missing_replay_asset_rejected_before_model_mutation(tmp_path):
    trainer = iterative_trainer(tmp_path / "run")
    stop_after(trainer, lambda progress: progress.get("iteration") == 2)
    with pytest.raises(Interrupted):
        trainer.train(checkpoint_dir=tmp_path / "checkpoints")
    checkpoint = tmp_path / "checkpoints" / "checkpoint-2"
    next((checkpoint / "assets" / "preference_replay").iterdir()).unlink()
    resumed = iterative_trainer(tmp_path / "new")
    before = resumed.agents[0].weight.detach().clone()
    with pytest.raises(FileNotFoundError):
        resumed.load_checkpoint(checkpoint)
    assert torch.equal(before, resumed.agents[0].weight)


def test_extra_state_rejects_arbitrary_attributes():
    trainer = toy_trainer()
    with pytest.raises(ValueError, match="extra-state"):
        trainer._restore_checkpoint_extra_state({"agents": []}, ".")
    assert len(trainer.agents) == 1


@pytest.mark.parametrize("scoring", ["reward_model", "task"])
def test_marlhf_iteration_restores_previous_reward_model_for_next_pairs(
    tmp_path, scoring
):
    seed(45)
    baseline = iterative_trainer(tmp_path / "baseline", MARLHFIterTrainer, scoring)
    baseline.train()
    expected_random = (random.random(), torch.rand(()).item())

    seed(45)
    interrupted = iterative_trainer(
        tmp_path / "interrupted", MARLHFIterTrainer, scoring
    )
    stop_after(interrupted, lambda progress: progress.get("iteration") == 2)
    with pytest.raises(Interrupted):
        interrupted.train(checkpoint_dir=tmp_path / "checkpoints")
    seed(91)
    resumed = iterative_trainer(tmp_path / "resumed", MARLHFIterTrainer, scoring)
    resumed.train(resume_from_checkpoint=tmp_path / "checkpoints" / "checkpoint-2")
    assert_same_training(baseline, resumed)
    if scoring == "reward_model":
        assert torch.equal(baseline.reward_model.weight, resumed.reward_model.weight)
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(
                baseline.reward_optimizer.state_dict()["state"][0][name],
                resumed.reward_optimizer.state_dict()["state"][0][name],
            )
    else:
        assert baseline.reward_model is resumed.reward_model is None
        assert baseline.reward_optimizer is resumed.reward_optimizer is None
    assert baseline.events == interrupted.events + resumed.events
    assert expected_random == (random.random(), torch.rand(()).item())


@pytest.mark.parametrize("value", [True, "1", 0.5, -1])
@pytest.mark.parametrize("kind", ["madpo", "marlhf", "madpo_iter"])
def test_preference_checkpoint_cursors_reject_noninteger_or_negative_values(
    tmp_path, kind, value
):
    if kind == "madpo":
        trainer = toy_trainer()
        trainer._checkpoint_progress = {
            "kind": kind,
            "epoch": value,
            "updates_seen": 0,
            "preference_pairs": [pair(1)],
            "complete": False,
        }
        run = trainer._train_madpo_checkpointed
    elif kind == "marlhf":
        trainer = marlhf_trainer()
        trainer._checkpoint_progress = {
            "kind": kind,
            "phase": "online_rl",
            "rl": {"kind": "magrpo", "epoch": value},
        }
        run = trainer._train_marlhf_checkpointed
    else:
        from comlrl.trainers.preference.checkpointing import (
            train_iterations_checkpointed,
        )

        trainer = iterative_trainer(tmp_path)
        trainer._checkpoint_progress = {
            "kind": kind,
            "iteration": value,
            "updates_seen": 0,
            "total_pairs": 0,
            "complete": False,
        }
        run = lambda: train_iterations_checkpointed(trainer, reward_model=False)
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize("kind", ["madpo", "marlhf", "madpo_iter"])
def test_preference_checkpoint_cursors_require_complete_shape(tmp_path, kind):
    if kind == "madpo":
        trainer = toy_trainer()
        run = trainer._train_madpo_checkpointed
    elif kind == "marlhf":
        trainer = marlhf_trainer()
        run = trainer._train_marlhf_checkpointed
    else:
        from comlrl.trainers.preference.checkpointing import (
            train_iterations_checkpointed,
        )

        trainer = iterative_trainer(tmp_path)
        run = lambda: train_iterations_checkpointed(trainer, reward_model=False)
    trainer._checkpoint_progress = {"kind": kind}
    with pytest.raises(ValueError):
        run()


def preflight_trainer(kind, directory):
    if kind == "madpo":
        return toy_trainer()
    if kind == "marlhf":
        return marlhf_trainer()
    return iterative_trainer(
        directory, MARLHFIterTrainer if kind == "marlhf_iter" else MADPOIterTrainer
    )


def saved_progress(kind):
    if kind == "madpo":
        return {
            "kind": kind,
            "epoch": 2,
            "updates_seen": 4,
            "preference_pairs": [pair(1)],
            "complete": False,
        }
    if kind == "marlhf":
        return {
            "kind": kind,
            "phase": "online_rl",
            "rl": {"kind": "magrpo", "epoch": 2},
        }
    return {
        "kind": kind,
        "iteration": 2,
        "updates_seen": 4,
        "total_pairs": 1,
        "complete": False,
    }


@pytest.mark.parametrize("kind", ["madpo", "marlhf", "madpo_iter", "marlhf_iter"])
@pytest.mark.parametrize("damage", ["malformed", "missing", "shortened"])
def test_progress_preflight_rejects_before_restoring_any_state(tmp_path, kind, damage):
    seed(101)
    source = preflight_trainer(kind, tmp_path / "source-run")
    source._update_from_preference_batch([pair(4)])
    source.env_step = 99
    source._checkpoint_progress = saved_progress(kind)
    if kind == "marlhf":
        source._init_reward_model()
    cursor = "iteration" if kind.endswith("_iter") else "epoch"
    cursor_progress = source._checkpoint_progress.get("rl", source._checkpoint_progress)
    if damage == "malformed":
        cursor_progress[cursor] = "2"
    elif damage == "missing":
        cursor_progress.pop(cursor)
    checkpoint = tmp_path / "checkpoint"
    source.save_checkpoint(checkpoint)

    seed(202)
    resumed = preflight_trainer(kind, tmp_path / "new-run")
    if damage == "shortened":
        if kind.endswith("_iter"):
            resumed.args.num_iterations = 1
        else:
            resumed.args.num_train_epochs = 1
    if kind == "marlhf":
        resumed._init_reward_model = lambda: pytest.fail("initialized before preflight")
    before_weights = resumed.agents[0].weight.detach().clone()
    before_torch = torch.get_rng_state().clone()
    before_python = random.getstate()
    with pytest.raises(ValueError):
        resumed.load_checkpoint(checkpoint)
    assert torch.equal(before_weights, resumed.agents[0].weight)
    assert not resumed.optimizers[0].state
    assert resumed.env_step == 0
    assert not hasattr(resumed, "_checkpoint_progress")
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    if kind == "marlhf":
        assert resumed.reward_model is None


def test_missing_preference_pairs_rejected_before_weight_restore(tmp_path):
    source = toy_trainer()
    source._checkpoint_progress = saved_progress("madpo")
    source._checkpoint_progress.pop("preference_pairs")
    source.save_checkpoint(tmp_path / "checkpoint")
    resumed = toy_trainer()
    before = resumed.agents[0].weight.detach().clone()
    with pytest.raises(ValueError, match="Incomplete"):
        resumed.load_checkpoint(tmp_path / "checkpoint")
    assert torch.equal(before, resumed.agents[0].weight)


@pytest.mark.parametrize("phase", ["online_rl", "complete"])
def test_marlhf_saved_rl_cursor_requires_reward_model_before_restore(tmp_path, phase):
    source = marlhf_trainer()
    source._checkpoint_progress = saved_progress("marlhf")
    source._checkpoint_progress["phase"] = phase
    source.save_checkpoint(tmp_path / "checkpoint")
    resumed = marlhf_trainer()
    before = resumed.agents[0].weight.detach().clone()
    with pytest.raises(ValueError, match="reward model"):
        resumed.load_checkpoint(tmp_path / "checkpoint")
    assert torch.equal(before, resumed.agents[0].weight)


def test_marlhf_saved_ac_cursor_requires_nested_snapshot_before_restore(tmp_path):
    source = marlhf_ac_trainer()
    source._init_reward_model()
    source._checkpoint_progress = {
        "kind": "marlhf",
        "phase": "online_rl",
        "rl": {"kind": "actor_critic", "epoch": 1},
    }
    source.save_checkpoint(tmp_path / "checkpoint")
    resumed = marlhf_ac_trainer()
    before = resumed.agents[0].weight.detach().clone()
    with pytest.raises(ValueError, match="actor-critic progress"):
        resumed.load_checkpoint(tmp_path / "checkpoint")
    assert torch.equal(before, resumed.agents[0].weight)
    assert resumed.reward_model is None
