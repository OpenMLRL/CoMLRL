---
title: Environment Transition
weight: 2
math: true
---

External feedback mechanisms control how environment observations are incorporated into prompts for subsequent turns. These mechanisms are usually implemented in CoMLRL's downstream environments.

{{< rollout-diagram diagram="environment-transition" animation="feedback" title="Feedback Loop" description="Agents combine their contributions into a response. Tool errors and user hints or new information are routed back to the corresponding agent for the next turn." >}}

## Custom External Feedback

Users can implement custom external feedback by defining a function with the following interface:

{{% hint info "Custom External Feedback Interface:" %}}
Custom External Feedback Interface:

- `prompt`: Original task prompt or problem description (required)
- `agent_completions`: List or tuple of completions from the previous turn, one per agent (required)
- `num_agents`: Number of agents in the system (required)
- `prompt_history_per_agent`: List containing one list of previous prompts per agent (optional)
- `response_history_per_agent`: List containing one list of previous responses per agent (optional)

The function must return a list or tuple of prompts for the next turn, one for each agent.
The trainer passes only the arguments above, with no additional keyword arguments. Users can configure mode-specific parameters through a closure or `functools.partial`.
{{% /hint %}}

By default, each returned prompt becomes the new `prompt` field and is passed through the corresponding agent's formatter.
If `external_prompt_passthrough=true`, the returned prompts are used directly without reformatting.
In MAGRPO, the external transition is called for each rollout branch with that branch's histories.

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
IAC/MAAC multi-turn training requires `num_generations` to be set to 1.
{{% /hint %}}

## Example Modes {#example-modes-expert-diagnosis-and-self-improvement}

An [environment for code generation](https://github.com/OpenMLRL/LLM_Collab_Code_Generation) includes three example external transition modes:

{{% hint success %}}

- `external.mode=expert_edits`: An external LLM (default: DeepSeek-Coder) proposes code edits. Follow-up prompts include edit suggestions and context from previous turns. With the required API keys, users can select a different expert (e.g., Claude, GPT) through `expert_model`.

- `external.mode=level_feedback`: Static AST checks and dynamic code execution provide a diagnosis. By default, the sandbox includes only the first test. The `sandbox_slice` parameter selects all tests (0, None, or 'all') or a specific number of tests (negative values are supported).

- `external.mode=plain`: This self-improvement mode includes only the prompts and responses from previous turns and a revision instruction.
{{% /hint %}}
