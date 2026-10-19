"""Trainers by family: reinforce (MAGRPO and its baselines), pbrl (preference-based RL) and
actor_critic. Every trainer runs in ``collaboration_mode="decentralized"`` (one actor per
agent) or ``"centralized"`` (one joint actor answering for all agents; see ``centralized``).
"""

from .actor_critic import (
    ActorCriticTrainerBase,
    IACConfig,
    IACTrainer,
    MAACConfig,
    MAACTrainer,
)
from .centralized import (
    CentralizedCollaboration,
    CentralizedComparatorAdapter,
    CentralizedComparatorParseError,
    TaggedCentralizedComparatorAdapter,
)
from .pbrl import (
    JointRewardModel,
    MADPOConfig,
    MADPOIterConfig,
    MADPOIterTrainer,
    MADPOTrainer,
    MARLHFConfig,
    MARLHFIterConfig,
    MARLHFIterTrainer,
    MARLHFTrainer,
)
from .reinforce import (
    MAGRPOConfig,
    MAGRPOTrainer,
    MAREINFORCEConfig,
    MAREINFORCETrainer,
    MAReMaxConfig,
    MAReMaxTrainer,
    MARLOOConfig,
    MARLOOTrainer,
)

__all__ = [
    "ActorCriticTrainerBase",
    "IACConfig",
    "IACTrainer",
    "MAACConfig",
    "MAACTrainer",
    "CentralizedCollaboration",
    "CentralizedComparatorAdapter",
    "CentralizedComparatorParseError",
    "TaggedCentralizedComparatorAdapter",
    "JointRewardModel",
    "MADPOConfig",
    "MADPOIterConfig",
    "MADPOIterTrainer",
    "MADPOTrainer",
    "MARLHFConfig",
    "MARLHFIterConfig",
    "MARLHFIterTrainer",
    "MARLHFTrainer",
    "MAGRPOConfig",
    "MAGRPOTrainer",
    "MAREINFORCEConfig",
    "MAREINFORCETrainer",
    "MAReMaxConfig",
    "MAReMaxTrainer",
    "MARLOOConfig",
    "MARLOOTrainer",
]
