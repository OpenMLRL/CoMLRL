---
title: Multi-Turn Training
linkTitle: Multi-Turn Training
weight: 4
math: true
---

Many complex problems cannot be solved in a single turn. Agents need to interact with the environment to obtain useful feedback from other models or tools involved in the system, enabling iterative refinement and exploration of multiple solution paths.

## Multi-Turn MAGRPO

MAGRPO in the multi-turn setting (**MAGRPO-MT**) forms a tree-structured rollout expansion where branches represent different joint responses ([TreeRPO](https://arxiv.org/abs/2506.05183)). In each episode, a task is sampled from the dataset to construct initial observations and histories for all agents. At each turn, agents generate a group of joint responses from their current observation-action history, with each response initiating a distinct rollout. Agents receive joint rewards for each response based on the accumulated history and current action. **Each rollout then evolves independently**, producing new joint observations as the environment dynamics unfold and spawning more rollouts at the next turn. This process continues until the terminal turn is reached.

### Joint Mode

MAGRPO supports two modes for forming joint responses at each turn:

- **Align**: Provides flexibility in the number of joint responses generated per turn, allowing any number of generations at each turn. However, generations are not fully utilized since only aligned responses across agents are combined. As training progresses over {{< katex inline=true >}}T{{< /katex >}} turns with {{< katex inline=true >}}N{{< /katex >}} agents, the total number of leaves grows as {{< katex inline=true >}}G^T{{< /katex >}}, where {{< katex inline=true >}}G{{< /katex >}} is the number of generations per turn.

- **Cross**: Maximizes the utilization of generations and provides more accurate value estimation with more samples by forming the Cartesian product of all agent responses. As training progresses over {{< katex inline=true >}}T{{< /katex >}} turns with {{< katex inline=true >}}N{{< /katex >}} agents, the total number of leaves grows as {{< katex inline=true >}}G^{N \cdot T}{{< /katex >}}, where each node has {{< katex inline=true >}}G^N{{< /katex >}} sibling joint actions.

{{% hint warning %}}
Note that only responses originating from the same rollout can be combined, as rollouts evolve independently.
{{% /hint %}}

### Rollout Tree Pruning

The rollout tree expands exponentially with the number of agents and turns, yielding {{< katex inline=true >}}G^{nt}{{< /katex >}} joint history-action values at step {{< katex inline=true >}}t{{< /katex >}}. While early termination (task completion or invalid responses) naturally limits tree growth, additional pruning strategies can further mitigate this exponential expansion.

A practical approach retains only the top-{{< katex inline=true >}}K{{< /katex >}} joint responses with highest rewards for each history, reducing total rollouts to {{< katex inline=true >}}K^t{{< /katex >}} at step {{< katex inline=true >}}t{{< /katex >}}. Additionally, homogeneous responses (e.g., identical code execution paths) from the same history can be merged to eliminate redundant nodes. Both top-{{< katex inline=true >}}K{{< /katex >}} filtering and uniqueness operations {{< katex inline=true >}}f{{< /katex >}} cause the behavior policy {{< katex inline=true >}}\boldsymbol{\pi}^f_{\boldsymbol{\theta}}{{< /katex >}} to diverge from the learning policy {{< katex inline=true >}}\boldsymbol{\pi}_{\boldsymbol{\theta}}{{< /katex >}}, requiring importance sampling ratios {{< katex inline=true >}}\rho^{(g)}_{i,t}=\frac{\pi_{\theta_i}(a^{(g)}_{i,t}|h_{i, t})}{\pi^f_{\theta_i}(a^{(g)}_{i,t}|h_{i, t})}{{< /katex >}} to correct the advantage estimates {{< katex inline=true >}}\widehat{A}(\mathbf{h}_t, \mathbf{a}^{(g)}_t) = R^\dagger(\mathbf{h}_t, \mathbf{a}^{(g)}_t) - \frac{1}{|\mathbf{a}^\mathcal{G}_t|}\sum_{\mathbf{a}^{(k)}_t \in f(\mathbf{a}^\mathcal{G}_t)} R^\dagger(\mathbf{h}_t, \mathbf{a}^{(k)}_t){{< /katex >}} and policy objectives. However, this approach primarily optimizes "good" responses that may not always be generated during testing.

Scalability can be further enhanced by training agents to learn a proximal policy {{< katex inline=true >}}\tilde{\boldsymbol{\pi}}(\cdot|\tilde{h}_t){{< /katex >}} based on either a recent history segment {{< katex inline=true >}}\mathbf{h}_{\tilde{t}:t}{{< /katex >}} or a latent history representation {{< katex inline=true >}}\tilde{\mathbf{h}}_{t}{{< /katex >}}. Pruning becomes more effective with shorter histories, as they are more likely to generate homogeneous responses. The importance sampling ratio becomes {{< katex inline=true >}}\tilde{\rho}^{(g)}_{i,t}=\frac{\tilde{\pi}_{\theta_i}(a^{(g)}_{i,t}|\tilde{h}_{i, t})}{\tilde{\pi}^f_{\theta_i}(a^{(g)}_{i,t}|\tilde{h}_{i, t})}{{< /katex >}}, where joint responses come from the proximal policy {{< katex inline=true >}}\mathbf{a}_{t}^{(g)}\gets \tilde{\boldsymbol{\pi}}(\cdot|\tilde{\mathbf{h}}_t){{< /katex >}}. While rolling out with the proximal policy enables more aggressive pruning of homogeneous responses, the ratio between the initial policy and proxy is intractable, necessitating backup of all old parameters.

## External Feedback

External feedback mechanisms control how environment observations are incorporated into prompts for subsequent turns. CoMLRL provides built-in external transition modes as examples, and users can define custom external feedback functions to suit their specific tasks.

### Built-in Modes

CoMLRL includes three example external transition modes for code generation tasks:

- **expert_edits**: Uses an external LLM (default: DeepSeek-Coder) to propose code edits. The follow-up prompts include edit suggestions with context from previous generations. This mode is configurable via `expert_model` to use different models (e.g., Claude, GPT) when API keys are available.

- **level_feedback**: Executes code against test cases and includes diagnostic feedback in the prompts. By default, includes the first test assertion; configurable via `sandbox_slice` to include all tests (0, None, or 'all'), specific number of tests, or last assertions (negative values).

- **plain**: Minimal feedback mode that includes previous responses and revision instructions without diagnostics or test results. Useful for tasks where external execution is not available or desired.

### Custom External Feedback

Users can implement custom external feedback by defining a function that takes the original prompt, agent completions from the previous turn, and optional histories, then returns formatted prompts for the next turn. The function signature should match:

```python
def custom_external(
    prompt: str,
    agent_completions: List[str],
    num_agents: int,
    prompt_history_per_agent: Optional[List[List[str]]] = None,
    response_history_per_agent: Optional[List[List[str]]] = None,
    **kwargs
) -> List[str]:
    # Custom logic to format next-turn prompts
    return next_turn_prompts
```

This allows full flexibility in how environment feedback, tool outputs, or other contextual information is integrated into the multi-turn training loop.
