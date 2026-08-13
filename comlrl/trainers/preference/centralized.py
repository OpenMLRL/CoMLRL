import re
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence


class CentralizedComparatorParseError(ValueError):
    """Raised when a centralized completion cannot be split into agent outputs."""


class CentralizedComparatorAdapter(Protocol):
    """Domain adapter for centralized comparator prompting and parsing."""

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        """Build one prompt that asks a policy to produce every agent output."""

    def parse_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        num_agents: int,
    ) -> Sequence[str]:
        """Split one centralized completion into agent-indexed outputs."""

    def build_sequential_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
        agent_index: int,
        previous_outputs: Sequence[str],
    ) -> str:
        """Build the prompt for one factor in a sequential joint action."""

    def parse_sequential_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        agent_index: int,
    ) -> str:
        """Extract one agent output from a sequential completion."""


@dataclass(frozen=True)
class TaggedCentralizedComparatorAdapter:
    """Domain-neutral centralized protocol using indexed XML-style tags."""

    def build_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
    ) -> str:
        del batch_item
        prompt_sections = "\n\n".join(
            f"Agent {agent_idx} original prompt:\n{prompt}"
            for agent_idx, prompt in enumerate(agent_prompts)
        )
        output_sections = "\n".join(
            f"<agent_{agent_idx}>\n...\n</agent_{agent_idx}>"
            for agent_idx in range(len(agent_prompts))
        )
        return f"""You are a centralized coordinator producing the separate outputs of {len(agent_prompts)} agents.

Each output must satisfy its corresponding original prompt. Keep the outputs separate;
do not merge them or add commentary outside the required tags.

{prompt_sections}

Return exactly one section for each agent in this format:
{output_sections}
"""

    def parse_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        num_agents: int,
    ) -> Sequence[str]:
        del batch_item
        outputs: List[str] = []
        found = False
        for agent_idx in range(num_agents):
            pattern = rf"<\s*agent_{agent_idx}\s*>(.*?)<\s*/\s*agent_{agent_idx}\s*>"
            match = re.search(
                pattern,
                str(completion),
                flags=re.IGNORECASE | re.DOTALL,
            )
            found = found or match is not None
            outputs.append(match.group(1).strip() if match is not None else "")
        if not found:
            raise CentralizedComparatorParseError(
                "Centralized completion did not contain any indexed agent sections."
            )
        return outputs

    def build_sequential_prompt(
        self,
        batch_item: Dict[str, Any],
        agent_prompts: Sequence[str],
        agent_index: int,
        previous_outputs: Sequence[str],
    ) -> str:
        del batch_item
        if agent_index < 0 or agent_index >= len(agent_prompts):
            raise ValueError("agent_index must identify one of the agent prompts.")
        if len(previous_outputs) != agent_index:
            raise ValueError(
                "previous_outputs must contain exactly the outputs generated before "
                "agent_index."
            )

        prompt_sections = "\n\n".join(
            f"Agent {idx} original prompt:\n{prompt}"
            for idx, prompt in enumerate(agent_prompts)
        )
        if previous_outputs:
            context = "\n\n".join(
                f"Final Agent {idx} output:\n<agent_{idx}>\n{output}\n</agent_{idx}>"
                for idx, output in enumerate(previous_outputs)
            )
        else:
            context = "No earlier agent output has been generated yet."

        return f"""You are Agent {agent_index} in a centralized sequential coordinator.

The team is producing one joint action. You can inspect every agent assignment and
all earlier finalized outputs. Produce only Agent {agent_index}'s factor of the joint
action, making it compatible with the earlier outputs without rewriting them.

{prompt_sections}

Earlier finalized outputs:
{context}

Return exactly this structure and no text outside it:
<agent_{agent_index}>
the complete Agent {agent_index} output
</agent_{agent_index}>
"""

    def parse_sequential_completion(
        self,
        completion: str,
        batch_item: Dict[str, Any],
        agent_index: int,
    ) -> str:
        del batch_item
        pattern = (
            rf"<\s*agent_{agent_index}\s*>(.*?)" rf"<\s*/\s*agent_{agent_index}\s*>"
        )
        match = re.search(
            pattern,
            str(completion),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise CentralizedComparatorParseError(
                "Sequential centralized completion did not contain the required "
                f"agent_{agent_index} section."
            )
        return match.group(1).strip()
