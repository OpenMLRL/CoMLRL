---
title: Contributing
weight: 2
---

Thanks for your interest in helping build CoMLRL! This guide walks you through reporting issues, contributing changes, and keeping the codebase healthy.

## Development

{{% steps %}}

1. Fork the upstream repository.

2. Git clone and synchronize with upstream.
    ```bash
      git clone https://github.com/<your-username>/CoMLRL.git
      cd CoMLRL
      git remote add upstream https://github.com/OpenMLRL/CoMLRL.git
      git fetch upstream
      git checkout -b feature/<short-description> upstream/main
      git fetch upstream && git rebase upstream/main
    ```

3. Implement new features or fix bugs, and updating documentation as needed.

4. Open pull requests to the upstream repository and wait for review.

{{% /steps %}}

## Sponsorship

It takes a lot of resources to fine-tune LLMs with MARL. We would appreciate any sponsorship to help with the resource costs. Please reach out to the maintainers for sponsorship opportunities.
