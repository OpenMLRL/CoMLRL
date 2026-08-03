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


def test_iterative_config_allows_generic_centralized_agent_counts():
    config = MADPOIterConfig(
        agent_devices="cpu",
        num_agents=3,
        comparator_generation_mode="centralized",
        comparator_centralized_agent_index=2,
    )
    assert config.num_agents == 3


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
