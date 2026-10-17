---
title: Environment Transition
weight: 2
math: true
---

External feedback turns environment observations into prompts for later turns. CoMLRL's downstream environments usually implement it.

{{< rollout-diagram diagram="environment-transition" direction="right" title="Environment transition" description="At turn t, the agents' responses enter the external transition, which returns each agent's own observation for turn t+1." >}}

## Custom External Feedback

Users can implement custom external feedback by defining a function with the following interface:

{{% hint info %}}
Custom External Feedback Interface:

- `prompt`: Original task prompt or problem description (required)
- `agent_completions`: List or tuple of completions from the previous turn, one per agent (required)
- `num_agents`: Number of agents in the system (required)
- `prompt_history_per_agent`: List containing one list of previous prompts per agent (optional)
- `response_history_per_agent`: List containing one list of previous responses per agent (optional)

The function returns a list or tuple with one next-turn prompt per agent. The trainer passes only the arguments above, so mode-specific parameters go through a closure or `functools.partial`.
{{% /hint %}}

By default, each returned prompt becomes the new `prompt` field and goes through the corresponding agent's formatter; with `external_prompt_passthrough=True`, the prompts are used as they are. In MAGRPO, the transition is called for each rollout branch with that branch's histories.

The following example constructs prompts for the next turn:

```python
from typing import List, Optional

def custom_external(
    prompt: str,
    agent_completions: List[str],
    num_agents: int,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
) -> List[str]:
    # Custom logic to format next-turn prompts
    # Access environment feedback, tool outputs, etc.
    next_turn_prompts = []
    for i in range(num_agents):
        # Format prompt for agent i based on history and feedback
        next_prompt = f"{prompt}\nPrevious attempt: {agent_completions[i]}\nPlease revise."
        next_turn_prompts.append(next_prompt)
    return next_turn_prompts
```

{{% hint warning %}}
Multi-turn IAC/MAAC training requires `num_generations=1`.
{{% /hint %}}

## Example Modes {#example-modes-expert-diagnosis-and-self-improvement}

An [environment for code generation](https://github.com/OpenMLRL/LLM_Collab_Code_Generation) includes three example external transition modes:

{{% hint success %}}

- `external.mode=expert_edits`: An external LLM (default: DeepSeek-Coder) proposes code edits. Follow-up prompts include edit suggestions and context from previous turns. With the required API keys, users can select a different expert (e.g., Claude, GPT) through `expert_model`.

- `external.mode=level_feedback`: Static AST checks and dynamic code execution provide a diagnosis. By default, the sandbox includes only the first test. The `sandbox_slice` parameter selects all tests (0, None, or 'all') or a specific number of tests (negative values are supported).

- `external.mode=plain`: This self-improvement mode includes only the prompts and responses from previous turns and a revision instruction.
{{% /hint %}}
