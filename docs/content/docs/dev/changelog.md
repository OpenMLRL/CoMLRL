---
title: Changelog
weight: 3
---

---

## Version 1.4.1

- Support comparator to be a centralized coordinator. ([#69](https://github.com/OpenMLRL/CoMLRL/pull/69)).
- Allow to deploy LLM agents, reward model and comparator in different devices.

## Version 1.4.0

- Add online preference-learning methods. ([#68](https://github.com/OpenMLRL/CoMLRL/pull/68)).
- Add dataset drift logger.

## Version 1.3.9

- Add offline preference-learning methods, MADPO, MARLHF, etc. ([#67](https://github.com/OpenMLRL/CoMLRL/pull/67)).

## Version 1.3.8

- Allow to use reference model for all algorithms (PR [#65](https://github.com/OpenMLRL/CoMLRL/pull/65)).
- Unify the logger between different trainers (PR [#66](https://github.com/OpenMLRL/CoMLRL/pull/66)).

## Version 1.3.7

- Remove the redundant sampling hyperparameters in algorithms.
- Allow multi-gpu training with MP (see results at PR [#62](https://github.com/OpenMLRL/CoMLRL/pull/62)).

## Version 1.3.6

- Fixed critical bug of loading heterogeneous models and reform the model loading logics (see results at PR [#60](https://github.com/OpenMLRL/CoMLRL/pull/60))
- Polish the docs.

## Version 1.3.5

- Add unit tests for hyperparameter constraints.
- Clean legacy interfaces.

## Version 1.3.4

- Fix the bug of loading heterogeneous models and reform the loading logics.
- Enable MBGD in MAGRPO to align with MAAC and IAC.
- Remove redundant and legacy hyperparameters (e.g., model kwargs, patching hyperparameters).
- Clean multi-device legacy, like drop last and num_workers.
- Add unit tests for model loading and separate it from CI as a badge.
- Clean short functions.
- Reorganize the docs and align the parameters.

## Version 1.3.3
- Compact MAREINFORCETrainer derivation, and move to the new folder.
- Unify the interface for different trainers.
- Remove redundant patches and wrappers.
- Reorganize the variables in the config yamls.

## Version 1.3.2

- Fix wandb logging issue in MAGRPOTrainer.

## Version 1.3.1

- Allow batch training in MAGRPOTrainer, IACTrainer and MAACTrainer.
- Allow multi-turn training in IACTrainer and MAACTrainer.
- Change the x-axis from data_step to env_step.

## Version 1.3.0

- Use TD error as critic update target in IACTrainer and MAACTrainer.

## Version 1.2.9

- Add MAACTrainer (separated centralized critic), now both IACTrainer and MAACTrainer can support single-turn training.

## Version 1.2.8

- The critic in IACTrainer now estimate V rather than Q.

## Version 1.2.7

- Change the IPPOTrainer to be IACTrainer.

## Version 1.2.6

- Including MAGRPO, MAREINFORCE, MARLOO, MAREMAX, and IPPO trainers for multi-agent reinforcement learning with LLMs.
- Support for multi-turn training with custom external feedback mechanisms.
- LLM collaboration environments for various tasks.
- Comprehensive documentation and examples for getting started.
