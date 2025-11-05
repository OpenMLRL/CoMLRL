# Contributing to CoMLRL

Thanks for your interest in helping shape CoMLRL! This guide walks you through reporting issues, contributing changes, and keeping the codebase healthy.

## Before You Start
- Review existing issues and discussions to avoid duplicating work.
- Be respectful and collaborative; assume good intent in reviews.
- For non-trivial changes, open an issue or discussion first so we can align on the scope.

## Reporting Issues

Provide as much context as possible:

- Environment (`python --version`, `torch --version`, GPU/CPU info).
- Exact commands and minimal repro cases.
- Logs, stack traces, or screenshots.

## Development Workflow

1. **Stay in sync with `main`**
    - You should fork and track the upstream repository:
        ```bash
        git clone https://github.com/<your-username>/CoMLRL.git
        cd CoMLRL
        git remote add upstream https://github.com/OpenMLRL/CoMLRL.git
        git fetch upstream
        git checkout -b feature/<short-description> upstream/main
        ```
     - Periodically resync with `git fetch upstream && git rebase upstream/main` (or `git merge upstream/main`) so your branch stays current.
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
