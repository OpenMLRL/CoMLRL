---
title: Environment Transition
weight: 2
math: true
---

External feedback mechanisms control how environment observations are incorporated into prompts for subsequent turns; they are usually implemented in CoMLRL's downstream environments.

## Custom External Feedback

Users can implement custom external feedback by defining a function with the following interface:

{{% hint info %}}
Custom External Feedback Interface:

- `prompt`: Original task prompt/problem description (required)
- `agent_completions`: List or tuple of completions from the previous turn, one per agent (required)
- `num_agents`: Number of agents in the system (required)
- `prompt_history_per_agent`: List of prompt histories for each agent, where each history is a list of prompts from previous turns (optional)
- `response_history_per_agent`: List of response histories for each agent, where each history is a list of responses from previous turns (optional)

The function must return a list or tuple of prompts for the next turn, one for each agent.
The trainer only passes the arguments above (no extra kwargs), so any mode-specific parameters should be captured via a closure or `functools.partial`.
{{% /hint %}}

By default, returned prompts are inserted as the new `prompt` field and then passed through each agent's formatter.
If `external_prompt_passthrough=true`, the returned prompts are used directly without reformatting.
In MAGRPO, the external transition is called per rollout branch with that branch's histories.

For example:

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
For IAC/MAAC multi-turn training, `num_generations` must be set to 1.
{{% /hint %}}

## Example Modes (Expert, Diagnosis, and Self-Improvement)

An [environment for code generation](https://github.com/OpenMLRL/LLM_Collab_Code_Generation) includes three example external transition modes:

{{% hint success %}}

- `external.mode=expert_edits`: Uses an external LLM (default: DeepSeek-Coder) to propose code edits. Follow-up prompts include edit suggestions with context from previous turns. It can be configured via `expert_model` for different experts (e.g., Claude, GPT) when API keys are available.

- `external.mode=level_feedback`: Performs static AST checks and dynamically executes code to provide a diagnosis. By default, the sandbox includes only the first test; use `sandbox_slice` to include all tests (0, None, or 'all') or a specific number of tests (negative values enabled).

- `external.mode=plain`: Self-improvement mode that just includes the prompts and responses from previous turns and a revision instruction.
{{% /hint %}}

<div id="centralized-preference-collaboration"></div>
<div id="api-comparators"></div>

{{% hint info %}}
For actor/comparator layouts and centralized preference training, see [Training Framework]({{< relref "/docs/mapl/training-framework#centralized-preference-collaboration" >}}).
Remote comparator configuration is covered in [API Comparators]({{< relref "/docs/mapl/training-framework#api-comparators" >}}).
{{% /hint %}}
