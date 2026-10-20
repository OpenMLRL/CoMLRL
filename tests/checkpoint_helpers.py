"""CPU-only trainer doubles shared by the checkpoint/resume tests."""

import copy
from pathlib import Path
import random
from types import SimpleNamespace

from datasets import Dataset
import numpy as np
import torch

from comlrl.trainers.pbrl import (
    MADPOIterTrainer,
    MADPOTrainer,
    MARLHFIterTrainer,
    MARLHFTrainer,
)
from comlrl.trainers.pbrl.madpo import AgentPreferenceTensors, PreferencePair
from comlrl.trainers.reinforce.magrpo import MAGRPOConfig, NodeSample
from comlrl.utils.checkpoint import TrainingCheckpointMixin

from helpers import assert_nested_equal


class Interrupted(RuntimeError):
    """Raised by a test to stop training right after a checkpoint commit."""


def stop_after(trainer, predicate):
    commit = trainer._checkpoint_commit

    def interrupt(progress, force=False):
        commit(progress, force=force)
        if predicate(progress):
            raise Interrupted

    trainer._checkpoint_commit = interrupt


class StatefulReward:
    def __init__(self):
        self.calls = 3

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return [1.0]

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


class TinyTrainer(TrainingCheckpointMixin):
    """Linear agents, critics, references and a reward model with live optimizers."""

    def __init__(self, *, width=3, prompts=("first", "second"), **config):
        self.args = MAGRPOConfig(
            num_agents=2,
            num_turns=1,
            agent_devices="cpu",
            reference_kl_enabled=True,
            **config,
        )
        self.num_agents = 2
        self.agent_devices = [torch.device("cpu")] * 2
        self.critic_devices = [torch.device("cpu")] * 2
        self.parallel_training = "none"
        self.agents = [torch.nn.Linear(width, 2) for _ in range(2)]
        self.critics = [torch.nn.Linear(width, 1) for _ in range(2)]
        self.reference_models = copy.deepcopy(self.agents)
        for model in self.reference_models:
            model.requires_grad_(False)
            model.eval()
        self.reward_model = torch.nn.Linear(width, 1)
        self.optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.01) for model in self.agents
        ]
        self.critic_optimizers = [
            torch.optim.AdamW(model.parameters(), lr=0.02) for model in self.critics
        ]
        self.reward_optimizer = torch.optim.AdamW(
            self.reward_model.parameters(), lr=0.03
        )
        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
            for optimizer in self.optimizers
        ]
        self.tokenizers = []
        self.train_dataset = Dataset.from_dict({"prompt": list(prompts)})
        self.eval_dataset = self.train_dataset
        self.env_step = 17
        self._last_train_log_step = 11
        self._comparator_rng_states = {"cpu": torch.get_rng_state().clone()}
        self.target_seed = 71
        self.comparator_seed = -71
        self.rollout_buffers = [
            [
                NodeSample(
                    agent_idx=0,
                    turn_idx=0,
                    completions_data={"completion_input_ids": torch.tensor([1, 2, 3])},
                    returns=[1.25, -0.5],
                    node_mean_reward=0.5,
                    node_mean_return=0.375,
                    node_env_step=16,
                )
            ],
            [],
        ]
        self.reward_func = StatefulReward()
        self._checkpoint_progress = {"epoch": 1, "next_item": 0}


def advance_models(trainer):
    """Take one optimizer and scheduler step on every TinyTrainer model."""
    inputs = torch.tensor([[0.2, -0.4, 0.7], [0.6, 0.1, -0.2]])
    models = trainer.agents + trainer.critics + [trainer.reward_model]
    optimizers = (
        trainer.optimizers + trainer.critic_optimizers + [trainer.reward_optimizer]
    )
    for model, optimizer in zip(models, optimizers):
        optimizer.zero_grad(set_to_none=True)
        model(inputs).square().mean().backward()
        optimizer.step()
    for scheduler in trainer.schedulers:
        scheduler.step()


def assert_models_equal(actual, expected):
    for name in ("agents", "critics", "reference_models"):
        for left, right in zip(getattr(actual, name), getattr(expected, name)):
            assert_nested_equal(left.state_dict(), right.state_dict())
    assert_nested_equal(
        actual.reward_model.state_dict(), expected.reward_model.state_dict()
    )


class FakeRun:
    def __init__(self, id):
        self.id, self.project, self.entity = id, "research", "team"
        self.finished = False

    def finish(self):
        self.finished = True


def attach_rng_consuming_logger(trainer, *, initialized):
    """Replace W&B setup with a logger that draws from every global generator."""
    trainer.wandb_config = {}
    trainer.wandb_initialized = False
    trainer.logging_initializations = 0

    def initialize():
        random.random()
        np.random.random()
        torch.rand(())
        trainer.logging_initializations += 1
        trainer.wandb_initialized = True

    trainer._init_wandb = initialize
    if initialized:
        initialize()  # Match ordinary eager constructor logging.
    return trainer


class TinyPolicy(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False)

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "weights.pt")


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
    """A preference trainer over six fixed pairs and one linear policy."""
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


def marlhf_trainer(cls=MARLHFTrainer):
    """A toy MARLHF trainer with a linear reward model and a MAGRPO online phase."""
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


def iterative_trainer(directory, cls=MADPOIterTrainer, scoring="reward_model"):
    """A four-iteration toy with history comparators and all-history replay."""
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


class TinyActor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, value):
        return self.model(value)


class TinyACTrainer(TrainingCheckpointMixin):
    """Stands in for the MAAC child trainer that MARLHF builds for its RL phase."""

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
    """A toy MARLHF trainer whose online phase is a nested MAAC trainer."""
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


def assert_same_training(left, right):
    assert torch.equal(left.agents[0].weight, right.agents[0].weight)
    assert left.env_step == right.env_step
    a = left.optimizers[0].state_dict()["state"]
    b = right.optimizers[0].state_dict()["state"]
    for index in a:
        for key in a[index]:
            assert torch.equal(a[index][key], b[index][key])
