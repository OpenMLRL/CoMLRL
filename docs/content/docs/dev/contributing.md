---
title: Contributing
weight: 2
---

Thanks for your interest in helping build CoMLRL! This guide walks you through reporting issues, contributing changes, and keeping the codebase healthy.

## Development Workflow

1.
2. **Implement your change**
   - Keep commits focused; document behaviour changes.
   - Update READMEs, examples, or tutorials when you alter user-facing workflows.
3. **Validate locally**
   - Run tests and pre-commit hooks before pushing.
   - For training scripts, run a smoke test (small dataset/few steps) and capture key metrics for your PR description.
4. **Open a pull request**
   - Reference related issues or discussions.
   - Summarize changes, note test evidence, and list follow-up items if any.
   - Expect collaborative review; feedback improves quality.


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

4. Open pull requests to the upstream repository.

{{% /steps %}}
