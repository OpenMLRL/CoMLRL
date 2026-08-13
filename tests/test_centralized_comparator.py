from types import SimpleNamespace

import pytest

from comlrl.trainers.preference.centralized import (
    CentralizedComparatorParseError,
    TaggedCentralizedComparatorAdapter,
)
from comlrl.trainers.preference.iterative import (
    MADPOIterConfig,
    MADPOIterTrainer,
)


class _StubAdapter:
    def build_prompt(self, batch_item, agent_prompts):
        return f"centralized:{batch_item['id']}:{'|'.join(agent_prompts)}"

    def parse_completion(self, completion, batch_item, num_agents):
        return [f"{completion}-agent-{idx}" for idx in range(num_agents)]


class _WrongSizeAdapter(_StubAdapter):
    def parse_completion(self, completion, batch_item, num_agents):
        return [completion]


class _SequentialAdapter(_StubAdapter):
    def build_sequential_prompt(
        self,
        batch_item,
        agent_prompts,
        agent_index,
        previous_outputs,
    ):
        return (
            f"item={batch_item['id']};agent={agent_index};"
            f"roles={'|'.join(agent_prompts)};previous={'|'.join(previous_outputs)}"
        )

    def parse_sequential_completion(self, completion, batch_item, agent_index):
        return f"{batch_item['id']}:{agent_index}:{completion}"


def test_tagged_adapter_is_domain_neutral_and_supports_multiple_agents():
    adapter = TaggedCentralizedComparatorAdapter()
    prompt = adapter.build_prompt(
        {"id": 1},
        ["first role", "second role", "third role"],
    )
    assert "Agent 0 original prompt" in prompt
    assert "<agent_2>" in prompt
    assert "auxiliary" not in prompt.lower()

    outputs = adapter.parse_completion(
        "<agent_0>a</agent_0><agent_1>b</agent_1><agent_2>c</agent_2>",
        {},
        3,
    )
    assert outputs == ["a", "b", "c"]


def test_tagged_adapter_rejects_output_without_agent_sections():
    with pytest.raises(CentralizedComparatorParseError):
        TaggedCentralizedComparatorAdapter().parse_completion("plain text", {}, 2)


def test_tagged_adapter_builds_candidate_specific_sequential_context():
    adapter = TaggedCentralizedComparatorAdapter()
    prompt = adapter.build_sequential_prompt(
        {},
        ["first role", "second role"],
        1,
        ["first output"],
    )

    assert "Agent 1 in a centralized sequential coordinator" in prompt
    assert "Final Agent 0 output" in prompt
    assert "first output" in prompt
    assert "<agent_1>" in prompt
    assert (
        adapter.parse_sequential_completion(
            "<agent_1>second output</agent_1>",
            {},
            1,
        )
        == "second output"
    )


def test_iterative_config_allows_generic_centralized_agent_counts():
    config = MADPOIterConfig(
        agent_devices="cpu",
        num_agents=3,
        comparator_generation_mode="centralized",
        comparator_centralized_agent_index=2,
    )
    assert config.num_agents == 3


def test_iterative_config_allows_sequential_centralized_generation():
    config = MADPOIterConfig(
        agent_devices="cpu",
        num_agents=3,
        comparator_generation_mode="centralized_sequential",
    )
    assert config.comparator_generation_mode == "centralized_sequential"


def test_iterative_config_rejects_single_agent_sequential_generation():
    with pytest.raises(ValueError, match="num_agents >= 2"):
        MADPOIterConfig(
            agent_devices="cpu",
            num_agents=1,
            comparator_generation_mode="centralized_sequential",
        )


def test_iterative_config_keeps_decentralized_as_default():
    config = MADPOIterConfig(agent_devices="cpu")
    assert config.comparator_generation_mode == "decentralized"


def test_centralized_split_preserves_candidate_and_agent_alignment():
    trainer = MADPOIterTrainer.__new__(MADPOIterTrainer)
    trainer.num_agents = 2
    trainer.centralized_comparator_adapter = _StubAdapter()

    outputs = trainer._split_centralized_comparator_outputs(
        ["candidate-0", "candidate-1"],
        batch_item={"id": "item"},
        prompt="centralized prompt",
    )

    assert outputs[0]["completions"] == [["candidate-0-agent-0", "candidate-1-agent-0"]]
    assert outputs[1]["completions"] == [["candidate-0-agent-1", "candidate-1-agent-1"]]


def test_centralized_split_validates_adapter_output_count():
    trainer = MADPOIterTrainer.__new__(MADPOIterTrainer)
    trainer.num_agents = 2
    trainer.centralized_comparator_adapter = _WrongSizeAdapter()

    with pytest.raises(ValueError, match="exactly 2 outputs"):
        trainer._split_centralized_comparator_outputs(
            ["candidate"],
            batch_item={"id": "item"},
            prompt="centralized prompt",
        )


def test_sequential_centralized_generation_preserves_candidate_context():
    trainer = MADPOIterTrainer.__new__(MADPOIterTrainer)
    trainer.num_agents = 2
    trainer.formatters = [
        lambda item: f"role-0:{item['id']}",
        lambda item: f"role-1:{item['id']}",
    ]
    trainer.centralized_comparator_adapter = _SequentialAdapter()
    stage_prompts = []

    def generate_stage(agent_idx, prompts):
        stage_prompts.append((agent_idx, list(prompts)))
        return [f"candidate-{idx}" for idx in range(len(prompts))]

    outputs = trainer._generate_sequential_centralized_outputs(
        {"id": "sample"},
        num_candidates=2,
        generate_stage=generate_stage,
    )

    assert outputs[0]["completions"] == [
        ["sample:0:candidate-0", "sample:0:candidate-1"]
    ]
    assert outputs[1]["completions"] == [
        ["sample:1:candidate-0", "sample:1:candidate-1"]
    ]
    assert "previous=sample:0:candidate-0" in stage_prompts[1][1][0]
    assert "previous=sample:0:candidate-1" in stage_prompts[1][1][1]


def test_decentralized_path_does_not_call_centralized_generation():
    trainer = MADPOIterTrainer.__new__(MADPOIterTrainer)
    trainer.args = SimpleNamespace(
        comparator_generation_mode="decentralized",
        comparator_policy="current",
    )
    trainer.agents = ["agent-0", "agent-1"]

    def fail(*args, **kwargs):
        raise AssertionError("centralized generation must not run")

    trainer._generate_centralized_comparator_outputs_for_item = fail
    trainer._generate_policy_outputs_for_item = (
        lambda policy_agents, batch_item, *, num_candidates, **kwargs: [
            policy_agents,
            batch_item,
            num_candidates,
        ]
    )

    output = trainer._generate_comparator_outputs_for_item(
        {"id": 7},
        iteration_idx=0,
        num_candidates=4,
    )
    assert output == [["agent-0", "agent-1"], {"id": 7}, 4]


def test_sequential_mode_does_not_use_existing_joint_centralized_path():
    trainer = MADPOIterTrainer.__new__(MADPOIterTrainer)
    trainer.args = SimpleNamespace(
        comparator_generation_mode="centralized_sequential",
        comparator_policy="current",
    )

    def fail(*args, **kwargs):
        raise AssertionError("single-generator centralized path must not run")

    trainer._generate_centralized_comparator_outputs_for_item = fail
    trainer._generate_sequential_centralized_comparator_outputs_for_item = (
        lambda batch_item, *, iteration_idx, num_candidates, **kwargs: [
            batch_item,
            iteration_idx,
            num_candidates,
        ]
    )

    output = trainer._generate_comparator_outputs_for_item(
        {"id": 8},
        iteration_idx=2,
        num_candidates=3,
    )
    assert output == [{"id": 8}, 2, 3]
