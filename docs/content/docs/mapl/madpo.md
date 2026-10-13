---
title: Multi-Agent DPO
weight: 3
math: true
---

Multi-Agent Direct Preference Optimization (MADPO) trains the agents directly on preferred and rejected joint responses, without a reward model or a separate RL stage. Its loss combines all agents' preference margins, so each policy learns from the team's outcome.

The training flow is: **task items → preference pairs → joint preference loss → policy updates**. [MARLHF]({{< relref "/docs/mapl/marlhf" >}}) uses the same pairs to fit a joint reward model first and then trains the policies with RL.

The diagram uses the same task item and preference pairs as the [MARLHF]({{< relref "/docs/mapl/marlhf" >}}) figure; here the agents' margins enter one joint loss directly.

{{< rollout-diagram diagram="madpo-training" title="MADPO training flow" description="The agents' margins on each preference pair add up to one joint loss that updates every policy directly." >}}

## Preference Pairs

Each agent {{< katex >}}i\in\mathcal{I}=\{1,\ldots,n\}{{< /katex >}} samples {{< katex >}}a_{i,t}\sim\pi_{\theta_i}(\cdot\mid h_{i,t}){{< /katex >}} from its local history; the responses form the joint action {{< katex >}}\mathbf{a}_t=\{a_{1,t},\ldots,a_{n,t}\}{{< /katex >}} with joint history {{< katex >}}\mathbf{h}_t{{< /katex >}}. Only a single turn is currently supported (horizon {{< katex >}}H=1{{< /katex >}}), so {{< katex >}}h_{i,0}{{< /katex >}} is agent {{< katex >}}i{{< /katex >}}'s prompt and {{< katex >}}\mathbf{a}_0{{< /katex >}} is the joint response.

For each task item in `train_dataset`, every agent samples `preference_num_candidates` responses from its formatted prompt. Responses with the same candidate index form a joint response and share one reward from `reward_func` (after the optional `reward_processor`). Of two joint responses with different rewards, the higher one is the winner {{< katex >}}w{{< /katex >}} and the other the loser {{< katex >}}l{{< /katex >}}; ties are discarded. Each pair {{< katex >}}(\mathbf{h}_H^{w},\mathbf{h}_H^{l}){{< /katex >}} holds the two joint histories of one task.

The base trainer collects the pairs once and reuses them for `num_train_epochs`; `pair_selection` and `preference_pairs_per_sample` set which pairs each task keeps.

{{% hint warning %}}
MADPO currently requires `num_turns=1` and `joint_mode="aligned"`. `pair_selection="all"` requires `preference_pairs_per_sample=None`. If no task yields a non-tied pair, training returns without updating the policies.
{{% /hint %}}

## Joint Preference Objective

Each agent's margin is its log-probability gap between the winning and losing responses, and the joint margin sums these over the agents:

{{< katex display=true >}}
\Delta_{\theta_i}=\sum_{t=0}^{H-1}\Big(\log\pi_{\theta_i}(a_{i,t}^{w}\mid h_{i,t}^{w})-\log\pi_{\theta_i}(a_{i,t}^{l}\mid h_{i,t}^{l})\Big),
\qquad
\Delta_{\boldsymbol{\theta}}=\sum_{i\in\mathcal{I}}\Delta_{\theta_i}.
{{< /katex >}}

MADPO minimizes

{{< katex display=true >}}
\mathcal{L}_{\mathrm{MADPO}}(\boldsymbol{\theta})
=-\mathbb{E}_{(\mathbf{h}_H^{w},\mathbf{h}_H^{l})\sim\mathcal{B}}
\Big[\log\sigma\big(\eta\,\Delta_{\boldsymbol{\theta}}\big)\Big],
{{< /katex >}}

where {{< katex >}}\sigma{{< /katex >}} is the sigmoid function, {{< katex >}}\eta>0{{< /katex >}} is `dpo_beta` (default 0.1), and {{< katex >}}\mathcal{B}{{< /katex >}} is the set of training pairs. The task reward only orders each pair; the loss itself uses the policies' log-probabilities.

The gradient for agent {{< katex >}}i{{< /katex >}} is

{{< katex display=true >}}
\nabla_{\theta_i}\mathcal{L}_{\mathrm{MADPO}}(\boldsymbol{\theta})
=-\eta\,\mathbb{E}_{\mathcal{B}}
\Big[\sigma\big(-\eta\,\Delta_{\boldsymbol{\theta}}\big)\,\nabla_{\theta_i}\Delta_{\theta_i}\Big].
{{< /katex >}}

The weight {{< katex >}}\sigma(-\eta\,\Delta_{\boldsymbol{\theta}}){{< /katex >}} depends on the whole team's margin, while the gradient flows only through agent {{< katex >}}i{{< /katex >}}'s own margin. In each batch, the trainer computes all margins once without gradients, then updates each agent with its own differentiable margin plus the others' detached margins.

### Reference-Free Implementation

Eq. 6 of the [MAPL paper](https://arxiv.org/abs/2609.32827) replaces {{< katex >}}\log\pi_{\theta_i}{{< /katex >}} in {{< katex >}}\Delta_{\theta_i}{{< /katex >}} with the log-ratio {{< katex >}}\log(\pi_{\theta_i}/\pi_{\mathrm{ref},i}){{< /katex >}} against a frozen reference policy. CoMLRL implements the reference-free form above: the inherited `reference_kl_enabled`, `reference_kl_coef`, and `reference_devices` settings do not add reference terms to the MADPO loss.

{{% hint warning %}}
The decentralized scorer drops the last completion token after trimming padding, so each agent's margin omits that token's log-probability. The centralized scorer covers the full joint completion.
{{% /hint %}}

## Configuration

Import `MADPOConfig` and `MADPOTrainer` from `comlrl.trainers.preference`.

{{% hint info %}}
**MADPOConfig** parameters:

- `num_agents`: Number of agents in decentralized mode, or task roles in centralized mode
- `collaboration_mode`: `"decentralized"` (default) or `"centralized"`
- `num_train_epochs`: Optimization epochs over the collected preference pairs
- `agent_learning_rate`: Policy learning rate
- `preference_num_candidates`: Candidates sampled per task and actor; at least 2
- `preference_pairs_per_sample`: Maximum pairs kept per task, or `None` for no limit
- `pair_selection`: `"reward_gap"` keeps the largest reward gaps, `"random"` a random subset, and `"all"` every non-tied pair
- `dpo_beta`: Positive coefficient on the joint margin
- `train_batch_size`: Preference pairs per update; defaults to `rollout_buffer_size`
- `max_new_tokens`, `temperature`, `top_p`, `top_k`: Candidate generation settings
- `parallel_training`, `agent_devices`: Policy execution mode and device placement
- `eval_interval`, `eval_num_samples`, `eval_batch_size`: Evaluation schedule and sample sizes; MADPO counts `eval_interval` in preference updates
- `logging_steps`: Training logging interval
{{% /hint %}}

{{% hint info %}}
**MADPOTrainer** setup:

- `agent_model` or `agents`: Model identifier for homogeneous agents, or a list of model identifiers or instances
- `num_agents`: Number of agents or task roles; must match `args.num_agents`
- `tokenizer`: One tokenizer, or one per actor
- `train_dataset`: Task items from which preference candidates are generated
- `eval_dataset`: Optional evaluation dataset
- `reward_func`: Required callable returning one shared reward per aligned joint response
- `reward_processor`: Optional post-processing of task rewards
- `formatters`: One prompt formatter shared by all roles, or a list of role-specific formatters
- `args`: `MADPOConfig` instance
- `model_config`: Optional model loading configuration
- `wandb_config`: Optional Weights & Biases configuration
- `eval_logger`, `eval_aggregator`: Optional evaluation metrics hooks, as in `MAGRPOTrainer`
- `centralized_comparator_adapter`: Prompt builder and response parser for centralized actors or comparators
{{% /hint %}}

Evaluation samples responses from the current policies and scores them with the task reward. The training metrics `turn_1/reward_mean` and `turn_1/expected_return` instead average the candidate rewards recorded when the pairs were collected; they do not reflect the updated policies.

## Collaboration and Iterative Training

Decentralized MADPO trains one policy per agent with the summed margin {{< katex >}}\Delta_{\boldsymbol{\theta}}{{< /katex >}}. With `collaboration_mode="centralized"`, one actor {{< katex >}}\boldsymbol{\pi}_{\boldsymbol{\theta}}(\mathbf{a}_t\mid\mathbf{h}_t){{< /katex >}} generates the complete joint response, and {{< katex >}}\Delta_{\boldsymbol{\theta}}{{< /katex >}} is that actor's margin on the joint text. Keep `num_agents` equal to the number of task roles, but supply one actor and tokenizer. The joint text is kept for preference training and parsed into role responses only for task rewards and evaluation. See [Training Framework]({{< relref "/docs/mapl/training-framework#collaboration-modes" >}}) for the layouts and adapter interface.

`MADPOIterTrainer` repeats data collection and training. Iteration {{< katex >}}\nu{{< /katex >}} pairs each new current-policy sample with the comparator sample at the same candidate index (either side can win), then applies the MADPO updates to the selected pairs {{< katex >}}\mathcal{B}^{(\nu)}{{< /katex >}}. An iteration is not a dialog turn, so the single-turn, aligned-response restriction still applies.

See [Training Framework]({{< relref "/docs/mapl/training-framework#iterative-training" >}}) for the iterative configuration, the comparator, and the replay buffer that selects {{< katex >}}\mathcal{B}^{(\nu)}{{< /katex >}}.
