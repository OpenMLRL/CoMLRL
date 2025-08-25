# CoMLRL

Cooperative Multi-LLM Reinforcement Learning.

## Setup

```bash
cd CoMLRL
conda create -n comlrl python=3.10 -y
conda activate comlrl
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
python examples/story-len-ratio.py
```

## Algorithms

- MAGRPO: Multi-Agent Group-Relative Policy Optimization