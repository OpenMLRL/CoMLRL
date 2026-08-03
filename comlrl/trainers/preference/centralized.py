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
