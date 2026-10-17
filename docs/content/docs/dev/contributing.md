---
title: Contributing
weight: 2
---

# Contributing

Contributions to CoMLRL are welcome. The steps below describe the development workflow.

## Development Guidelines

{{% steps %}}
1. Fork the upstream repository.
2. Clone the fork and keep it synchronized with upstream:
    ```bash
      git clone https://github.com/<your-username>/CoMLRL.git
      cd CoMLRL
      git remote add upstream https://github.com/OpenMLRL/CoMLRL.git
      git fetch upstream
      git checkout -b feature/<short-description> upstream/main
      git fetch upstream && git rebase upstream/main
    ```
3. Implement the feature or fix, and update the documentation as needed.
4. Open a pull request against the upstream repository for review.
{{% /steps %}}
