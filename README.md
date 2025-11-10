# <img src="docs/assets/comlrl.png" width="400px;" alt=""/>

[![OpenMLRL](https://img.shields.io/badge/OpenMLRL-Project-blue.svg)](https://openmlrl.github.io)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-CoMLRL-yellow.svg?logo=huggingface&logoColor=darkgrey)](https://huggingface.co/CoMLRL)
[![arXiv](https://img.shields.io/badge/arXiv-2508.04652-b31b1b.svg?logo=arxiv&logoColor=darkgrey)](https://arxiv.org/pdf/2508.04652)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=darkgrey)](https://www.python.org/downloads/release/python-3100/)
[![CI](https://github.com/OpenMLRL/CoMLRL/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenMLRL/CoMLRL/actions/workflows/ci.yml)
[![pre-commit.ci](https://github.com/OpenMLRL/CoMLRL/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/OpenMLRL/CoMLRL/actions/workflows/pre-commit.yml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

**Co**operative **M**ulti-**L**LM **R**einforcement **L**earning (**CoMLRL**) is a open-source library for training multiple LLMs to collaborate using Multi-Agent Reinforcement Learning (MARL). It provides implementations of various MARL algorithms for LLM collaboration and support for different environments and benchmarks.

## Installation

You can install the stable version of this library from PyPI using pip:

```bash
python3 -m pip install comlrl
```

If you want to use the latest version of CoMLRL, please clone this repository and install it in editable mode:

```bash
cd CoMLRL
pip install -r requirements.txt
pip install -e .
```

<em><sub>Note: Make sure you have a compatible `torch` installed according to the CUDA.</sub></em>

## Features

- **MARL trainers to optimize LLM collaboration:**
  - **_Multi-Agent REINFORCE_:** Critic-free policy gradient methods, including [MAREINFROCE](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/mareinforce.py), [MAGRPO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/magrpo.py), [MARLOO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/marloo.py), [MAREMAX](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/maremax.py).
    - Aligned individual response joint with `joint_mode='align'`.
    - Memory-efficient cross joint with `joint_mode='cross'`.
  - **_Multi-Agent PPO:_** Critic-based policy gradient methods, including [IPPO](https://github.com/OpenMLRL/CoMLRL/blob/main/comlrl/trainers/ippo.py).
    - Canonical IPPO with a separate critic with `use_separate_critic=True`.
    - Memory-efficient critic with value-head over actor with `use_separate_critic=False`.

- **Environments that simulate real-world tasks for training and evaluating LLM collaboration:**
  - [**_Writing Collaboration_**](https://github.com/OpenMLRL/LLM_Collab_Writing): Multiple LLM agents collaborate on processing articles.
    - [TLDR](https://huggingface.co/datasets/trl-lib/tldr) - Summarizing Reddit posts.
    - [ArXiv](http://arxiv.org/abs/1905.00075) - Expanding abstracts into introductions.
  - [**_Code Generation_**](https://github.com/OpenMLRL/LLM_Collab_Code_Generation): Generate code solutions for programming problems.
    - [MBPP](https://arxiv.org/abs/2108.07732) - Mostly basic python problems.
    - [HumanEval](https://arxiv.org/abs/2107.03374) - Handwritten evaluation problems
    - [CoopHumanEval](https://huggingface.co/datasets/OpenMLRL/CoopHumanEval) - HumanEval with cooperative nature.
  - [**_Code Completion_**](https://github.com/OpenMLRL/LLM_Collab_Code_Completion): Complete code snippets based on given contexts.
    - [ClassEval](https://conf.researchr.org/details/icse-2024/icse-2024-research-track/219/Evaluating-Large-Language-Models-in-Class-Level-Code-Generation) - Complete class-level code based on method stubs and docstrings.


<img src="docs/assets/demo.gif" width="600px;" alt=""/>

## Usage

Quick start by training 2 `Qwen-2.5` agents to summarize Reddit posts with MAGRPO:

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer

# Load dataset and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
dataset = load_dataset("trl-lib/tldr", split="train").select(range(128))

# Initialize trainer and start training
trainer = MAGRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    num_agents=2,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_func=lambda a, b: [abs(max(len(b[0]), 1) / max(len(a[0]), 1) - 3.0)],
    formatters=[lambda example: example["prompt"]] * 2,
    args=MAGRPOConfig(
        per_device_train_batch_size=1,
    ),
)
trainer.train()
```

## Contributing

We welcome contributions from the community! Please see [contributing guidelines](./CONTRIBUTING.md) on setting up a development environment and contribute.

Thanks to the gracious help of contributors:
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/LovelyBuggies"><img src="https://avatars.githubusercontent.com/u/29083689?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Shuo Liu</b></sub></a><br /><a href="#ideas" title="Ideas">🤔</a> <a href="https://github.com/OpenMLRL/LLM_Collab_Code_Generation" title="Maintain">🚧</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=LovelyBuggies" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a></td>
    <td align="center"><a href="https://github.com/Tenshi0x0"><img src="https://avatars.githubusercontent.com/u/105730496?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Tianle Chen</b></sub></a><br /><a href="https://github.com/OpenMLRL/LLM_Collab_Code_Completion" title="Maintain">🚧</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=Tenshi0x0" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3ATenshi0x0" title="Bug Report">🐛</a></td>
<td align="center"><a href="https://github.com/ryankamiri"><img src="https://avatars.githubusercontent.com/u/44690200?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Ryan Amiri</b></sub></a><br /><a href="https://github.com/OpenMLRL/LLM_Collab_Software_Engineering" title="Maintain">🚧</a> <a href="https://github.com/OpenMLRL/CoMLRL/commits?author=ryankamiri" title="Code">💻</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3Aryankamiri" title="Bug Report">🐛</a> </td>
    <td align="center"><a href="https://github.com/zedyelllion"><img src="https://avatars.githubusercontent.com/u/111674669?v=4?s=80" width="80px;" alt=""/><br /><sub><b>Zeyu Liang</b></sub></a><br /> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs"></a> <a href="https://github.com/OpenMLRL/CoMLRL/" title="Docs">📖</a> <a href="https://github.com/OpenMLRL/CoMLRL/issues?q=author%3Azedyelllion" title="Bug Report">🐛</a></td>
 </tr>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->
<sub>🤔: Foundational Ideas; 🚧: Maintenance; 💻: Code; 📖: Documentation; 🐛: Bug Report.</sub>

## Citation

Please cite our paper if you find this library useful in your research:

```bibtex
@misc{liu2025comlrl,
      title={LLM Collaboration With Multi-Agent Reinforcement Learning},
      author={Shuo Liu and Tianle Chen and Zeyu Liang and Xueguang Lyu and Christopher Amato},
      year={2025},
      eprint={2508.04652},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.04652},
}
```
