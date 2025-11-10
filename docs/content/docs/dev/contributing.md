---
title: Contributing
weight: 2
---

Thanks for your interest in helping build CoMLRL! This guide walks you through reporting issues, contributing changes, and keeping the codebase healthy.

## Development Guidelines

- Fork the upstream repository.
- Clone your fork and synchronize with upstream:
    ```bash
      git clone https://github.com/<your-username>/CoMLRL.git
      cd CoMLRL
      git remote add upstream https://github.com/OpenMLRL/CoMLRL.git
      git fetch upstream
      git checkout -b feature/<short-description> upstream/main
      git fetch upstream && git rebase upstream/main
    ```
- Implement new features or fix bugs, updating documentation as needed.
- Open a pull request to the upstream repository and wait for review.
