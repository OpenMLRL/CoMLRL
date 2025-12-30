---
title: Changelog
weight: 3
---

---

## Version 1.3.1

- Allow batch training in MAGRPOTrainer, IACTrainer and MAACTrainer
- Allow multi-turn training in IACTrainer and MAACTrainer
- Change the x-axis from data_step to env_step

## Version 1.3.0

Use TD error as critic update target in IACTrainer and MAACTrainer.

## Version 1.2.9

Add MAACTrainer (separated centralized critic), now both IACTrainer and MAACTrainer can support single-turn training.

## Version 1.2.8

The critic in IACTrainer now estimate V rather than Q.

## Version 1.2.7

Change the IPPOTrainer to be IACTrainer.

## Version 1.2.6

The first release of CoMLRL:

- Including MAGRPO, MAREINFORCE, MARLOO, MAREMAX, and IPPO trainers for multi-agent reinforcement learning with LLMs.
- Support for multi-turn training with custom external feedback mechanisms.
- LLM collaboration environments for various tasks.
- Comprehensive documentation and examples for getting started.
