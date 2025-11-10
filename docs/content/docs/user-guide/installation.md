---
title: Installation
weight: 1
---

You can create an venv or conda environment with Python 3.10+ and install CoMLRL as follows.

To install the stable version of this library from PyPI using pip:

```bash
python3 -m pip install comlrl
```

To access the latest features of CoMLRL, please clone this repository and install it in editable mode:

```bash
cd CoMLRL
pip install -r requirements.txt
pip install -e .
```

{{% hint warning %}}
Please make sure you have compatible `torch` installed according to your CUDA.
{{% /hint %}}
