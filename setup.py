from pathlib import Path

from setuptools import find_packages, setup

# Core dependencies (excluding PyTorch which should be installed separately)
core_requires = [
    "datasets>=2.0.0",
    "numpy>=1.21.0",
    "openai>=1.0.0",
    "packaging>=21.0",
    "pyyaml>=6.0.1",
    "tqdm>=4.65.0",
    "transformers>=4.30.0",
    "wandb>=0.15.0",
]

extras_require = {}

short_description = (
    "Open-Source Library for Cooperative Multi-LLM Reinforcement Learning"
)

long_description = short_description

setup(
    name="comlrl",
    use_scm_version=True,
    setup_requires=["setuptools-scm"],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=core_requires,
    extras_require=extras_require,
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
