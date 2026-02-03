from dataclasses import dataclass

from .magrpo import MAGRPOConfig, MAGRPOTrainer


@dataclass
class MAREINFORCEConfig(MAGRPOConfig):
    """
    Configuration for MAREINFORCE training.

    Inherits all settings from MAGRPOConfig; behavior is identical to MAGRPO
    except for the advantage computation, which uses plain returns with no
    baseline (REINFORCE without a baseline).
    """

    advantage_mode: str = "raw"


class MAREINFORCETrainer(MAGRPOTrainer):
    """
    Multi-Agent REINFORCE (no baseline) Trainer.

    Identical to MAGRPOTrainer except the advantage equals the raw return per
    generation (no group mean, no leave-one-out, no max).
    The resulting return per generation is applied uniformly to each agent,
    same as in MAGRPOTrainer.
    """

    default_config_cls = MAREINFORCEConfig
    algorithm_name = "MAREINFORCE"
