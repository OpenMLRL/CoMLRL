from dataclasses import dataclass

from .magrpo import MAGRPOConfig, MAGRPOTrainer


@dataclass
class MARLOOConfig(MAGRPOConfig):
    """
    Configuration for MARLOO training.

    Inherits all settings from MAGRPOConfig; behavior is identical to MAGRPO
    except for the advantage computation, which uses a return leave-one-out
    (RLOO) baseline across generations.
    """

    advantage_mode: str = "rloo"


class MARLOOTrainer(MAGRPOTrainer):
    """
    Multi-Agent Return Leave-One-Out (MARLOO) Trainer.

    Identical to MAGRPOTrainer except the advantage is computed with a
    leave-one-out mean over generations.
    The resulting advantage per generation is applied uniformly to each agent,
    same as in MAGRPOTrainer.
    """

    default_config_cls = MARLOOConfig
    algorithm_name = "MARLOO"
