# Contributing to CoMLRL

Thanks for your interest in helping shape Cooperative Multi-LLM Reinforcement Learning! This guide walks you through contributing changes, reporting issues, and keeping the codebase healthy.

## Local Development Setup
```bash
git clone https://github.com/OpenMLRL/CoMLRL.git
cd CoMLRL
conda create -n comlrl python=3.10
pip install -r requirements.txt
pip install -e .
pre-commit install
```

## Contribution Workflow
1. **Fork & branch**
   ```bash
   git checkout -b feature/<short-description>
   ```
2. **Implement your change**
   - Keep commits focused and descriptive.
   - Update documentation and examples when APIs or behaviour change.
3. **Validate locally**
   - Run the tests and pre-commit hooks listed above.
   - For training scripts, run a lightweight smoke test (e.g., small dataset or few steps) and summarize the outcome in your PR description.
4. **Open a pull request**
   - Reference related issues.
   - Provide a concise summary, test evidence, and any follow-up TODOs.
   - Expect feedback—reviews are collaborative and meant to improve quality.

## Documentation Contributions
- Fixing typos or clarifying explanations is always welcome.
- For larger guides or tutorials, include rendered screenshots or logs when relevant.
- Ensure README badges or links remain accurate—update them alongside code changes.

## Reporting Issues
When filing a bug or feature request, include:
- Environment details (`python --version`, `torch` version, GPU/CPU info).
- Exact commands used and minimal reproducible snippets or configs.
- Logs, stack traces, or screenshots if available.

The more context, the faster we can help.

## Citing CoMLRL
If your research or product builds on this project, see `CITATION.cff` or the README’s citation section for proper attribution.

We appreciate your help—thank you for building CoMLRL with us! ❤️
