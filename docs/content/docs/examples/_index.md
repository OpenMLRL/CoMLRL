---
title: Examples
weight: 1
bookCollapseSection: false
bookFlatSection: true
---

This section presents a complete two-agent training run and examples of more advanced usage.

## Quick Start

<div class="docs-resource-grid">
{{< resource-card page="docs/examples/quick-start/length-ratio-training" label="Tutorial" title="Length-Ratio Training" >}}
A two-agent training example with a length-ratio reward, covering the dataset, reward function, configuration, and training loop.
{{< /resource-card >}}
</div>

## Advanced Usage

<div class="docs-resource-grid">
{{< resource-card page="docs/examples/advanced/resume-training" label="Checkpoints" title="Resuming Training" >}}
Resuming an interrupted run or extending a finished one from training checkpoints, with all segments logged to one W&B run.
{{< /resource-card >}}
{{< resource-card page="docs/examples/advanced/multi-turn-feedback" label="Multi-turn" title="Multi-Turn Feedback" >}}
Two-turn training in which an external transition reports each agent's response length before the agents revise.
{{< /resource-card >}}
</div>

## Environments

Richer examples can be found in the [environments]({{< relref "docs/env" >}}), which apply CoMLRL to writing, coding, travel planning, and tool use.
