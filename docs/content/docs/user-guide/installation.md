---
title: Installation
weight: 1
---

CoMLRL provides several different ways for installation.
Python 3.10 or later is required. Install a PyTorch build compatible with your machine separately; it is intentionally excluded from the package's core dependencies.

These docs describe the current source checkout. Published [PyPI](https://pypi.org/project/comlrl/) and [conda-forge](https://anaconda.org/conda-forge/comlrl) packages can lag behind it, so use the source installation for features added on the `clean-up` branch.

## Install from PyPI

```bash
pip install comlrl
# install PyTorch compatible with your device
```

## Install from conda-forge

```bash
conda install -c conda-forge comlrl
# install PyTorch compatible with your device
```

## Install from source

To access the latest features of CoMLRL or to develop CoMLRL, clone this repository and install in editable mode:

```bash
git clone https://github.com/OpenMLRL/CoMLRL.git
cd CoMLRL
# To follow the version covered by these branch docs:
git checkout clean-up
pip install -e .
# install PyTorch compatible with your device
```
