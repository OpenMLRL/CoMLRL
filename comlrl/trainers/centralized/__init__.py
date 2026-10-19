"""Pieces shared by every trainer in centralized collaboration mode (one joint actor)."""

from .adapters import (
    CentralizedComparatorAdapter,
    CentralizedComparatorParseError,
    TaggedCentralizedComparatorAdapter,
)
from .collaboration import (
    CentralizedCollaboration,
    joint_policy_gradient_loss,
    joint_sequence_log_prob,
)
from .mode import (
    COLLABORATION_MODES,
    CollaborationModeMixin,
    validate_collaboration_mode,
)

__all__ = [
    "COLLABORATION_MODES",
    "CentralizedCollaboration",
    "CentralizedComparatorAdapter",
    "CentralizedComparatorParseError",
    "CollaborationModeMixin",
    "TaggedCentralizedComparatorAdapter",
    "joint_policy_gradient_loss",
    "joint_sequence_log_prob",
    "validate_collaboration_mode",
]
