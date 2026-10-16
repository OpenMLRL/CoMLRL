---
title: Contributing
weight: 2
---

# Contributing

Contributions to CoMLRL are welcome. This guide covers issue reporting, code changes, and codebase maintenance.

## Development Guidelines

{{% steps %}}
1. Contributors can fork the upstream repository.
2. The following commands clone the fork and synchronize it with upstream:
    ```bash
      git clone https://github.com/<your-username>/CoMLRL.git
      cd CoMLRL
      git remote add upstream https://github.com/OpenMLRL/CoMLRL.git
      git fetch upstream
      git checkout -b feature/<short-description> upstream/main
      git fetch upstream && git rebase upstream/main
    ```
3. Contributors can implement new features or fix bugs and update the documentation as needed.
4. Changes can be submitted to the upstream repository as a pull request for review.
{{% /steps %}}
