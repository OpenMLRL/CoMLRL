# <img src="docs/assets/comlrl.png" width="500px;" alt=""/>

[![OpenMLRL](https://img.shields.io/badge/OpenMLRL-Project-blue.svg)](https://openmlrl.github.io)
[![Hugging Face](https://img.shields.io/badge/huggingface-CoMLRL-yellow.svg)](https://huggingface.co/CoMLRL)
[![arXiv](https://img.shields.io/badge/arXiv-2508.04652-b31b1b.svg)](https://arxiv.org/pdf/2508.04652)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CI](https://github.com/OpenMLRL/CoMLRL/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenMLRL/CoMLRL/actions/workflows/ci.yml)
[![pre-commit.ci](https://github.com/OpenMLRL/CoMLRL/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/OpenMLRL/CoMLRL/actions/workflows/pre-commit.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning (**CoMLRL**) is a open-source library for training multiple LLMs to collaborate using Multi-Agent Reinforcement Learning (MARL). It provides implementations of various MARL algorithms for LLM collaboration in Multi-Agent Systems (MAS) and support for different environments and benchmarks.

<img src="docs/assets/dec-pomdp.png" width="600px;" alt=""/>

## Installation

To install a stable version from PyPI:

```bash
pip install comlrl # ensure torch is compatible
```

To use the latest version of CoMLRL, clone the repository and install the required dependencies:

```bash
cd CoMLRL
pip install -r requirements.txt  # ensure torch is compatible
pip install -e .
```

## Usage

See scripts in `examples/` for usage examples.

## Trainers

- **MAREINFORCE:** Multi-Agent REINFORCE without a baseline.

$$
  J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
  \Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} R^{(g)}_t \cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}|h_{i,t})\Bigg];
$$

- **MAGRPO:** Multi-Agent Group-Relative Policy Optimization, credits to [GRPO](https://arxiv.org/pdf/2402.03300),[Dr. GRPO](https://arxiv.org/abs/2503.20783), and [TreeRPO](https://arxiv.org/abs/2506.05183).

$$
  J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
  \Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big(R^{(g)}_t - \text{mean}(R^{\mathcal{G}}_t)\Big)\cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}|h_{i,t})\Bigg];
$$

- **MARLOO:** Multi-Agent REINFORCE Leave-One-Out, credits to [RLOO](https://openreview.net/forum?id=r1lgTGL5DE) and [Revisiting REINFORCE](https://arxiv.org/abs/2402.14740).

$$
  J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
  \Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big(R^{(g)}_t - \sum_{k\in \mathcal{G}, k\neq g}\frac{R^{(k)}_t}{|\mathcal{G}|-1}\Big)\cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}|h_{i,t})\Bigg];
$$

- **MAReMax:** Multi-Agent REINFORCE with Group Max, credits to [ReMax](https://arxiv.org/abs/2310.10505).

$$
  J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^\mathcal{G} \sim \mathbf{\pi}_{\mathbf{\theta}}}
  \Bigg[\frac{1}{|\mathcal{G}|}\sum_{g \in \mathcal{G}} \Big(R^{(g)}_t - \max(R_t^{\mathcal{G}}) \Big)\cdot \log \pi_{\theta_i}(a^{(g)}_{i,t}|h_{i,t})\Bigg];
$$

- **IPPO:** Independent PPO with parameter sharing between actor and critic for single-agent, single-turn fine-tuning.

- More algs are coming soon!

## Environments

This library supports LLM collaboration in various environments:

- [Writing Collaboration](https://github.com/OpenMLRL/LLM_Collab_Writing): Multiple LLM agents collaborate on processing articles.
  - [TLDR](https://huggingface.co/datasets/trl-lib/tldr) - Summarizing Reddit posts.
  - [ArXiv](http://arxiv.org/abs/1905.00075) - Expanding abstracts into introductions.
- [Code Generation](https://github.com/OpenMLRL/LLM_Collab_Code_Generation): Generate code solutions for programming problems.
  - [MBPP](https://arxiv.org/abs/2108.07732) - Mostly basic python problems.
  - [HumanEval](https://arxiv.org/abs/2107.03374) - Handwritten evaluation problems
  - [CoopHumanEval](https://huggingface.co/datasets/OpenMLRL/CoopHumanEval) - HumanEval with cooperative nature.
- [Code Completion](https://github.com/OpenMLRL/LLM_Collab_Code_Completion): Complete code snippets based on given contexts.
  - [ClassEval](https://conf.researchr.org/details/icse-2024/icse-2024-research-track/219/Evaluating-Large-Language-Models-in-Class-Level-Code-Generation) - Complete class-level code based on method stubs and docstrings.

## Acknowledgements

Thanks especially to the gracious help of contributors:
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/LovelyBuggies"><img src="https://avatars.githubusercontent.com/u/29083689?v=4?s=80" width="66px;" alt=""/><br /><sub><b>Shuo Liu</b></sub></a><br /><a href="#ideas" title="Ideas">🤔</a> <a href="https://github.com/OpenMLRL/LLM_Collab_Code_Generation" title="Maintain">🚧</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=LovelyBuggies" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a></td>
    <td align="center"><a href="https://github.com/Tenshi0x0"><img src="https://avatars.githubusercontent.com/u/105730496?v=4?s=80" width="66px;" alt=""/><br /><sub><b>Tianle Chen</b></sub></a><br /><a href="https://github.com/OpenMLRL/LLM_Collab_Code_Completion" title="Maintain">🚧</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=Tenshi0x0" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3ATenshi0x0" title="Bug Report">🐛</a></td>
<td align="center"><a href="https://github.com/ryankamiri"><img src="https://avatars.githubusercontent.com/u/44690200?v=4?s=80" width="66px;" alt=""/><br /><sub><b>Ryan Amiri</b></sub></a><br /><a href="https://github.com/OpenMLRL/LLM_Collab_Software_Engineering" title="Maintain">🚧</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=ryankamiri" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3Aryankamiri" title="Bug Report">🐛</a> </td>
    <td align="center"><a href="https://github.com/zedyelllion"><img src="https://avatars.githubusercontent.com/u/111674669?v=4?s=80" width="66px;" alt=""/><br /><sub><b>Zeyu Liang</b></sub></a><br /> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs"></a> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3Azedyelllion" title="Bug Report">🐛</a></td>
 </tr>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->
🤔 - Foundational Ideas; 🚧 - Maintenance; 💻 -  Code; 📖 - Documentation; 🐛 - Bug Report.
