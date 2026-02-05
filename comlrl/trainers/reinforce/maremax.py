from dataclasses import dataclass

from .magrpo import MAGRPOConfig, MAGRPOTrainer


@dataclass
class MAReMaxConfig(MAGRPOConfig):
    """
    Configuration for MAReMax training.

    Inherits all settings from MAGRPOConfig; behavior is identical to MAGRPO
    except for the advantage computation, which uses a max-baseline across
    generations: A_g = R_g - max(R_1..R_G).
    """

    advantage_mode: str = "max"


class MAReMaxTrainer(MAGRPOTrainer):
    """
    Multi-Agent Return Max-Baseline (MAReMax) Trainer.

    Identical to MAGRPOTrainer except the advantage is computed with a
    max baseline over generations:

        A_g = R_g - max_k R_k

    The resulting advantage per generation is applied uniformly to each agent,
    same as in MAGRPOTrainer.
    """

    default_config_cls = MAReMaxConfig
    algorithm_name = "MAReMax"
