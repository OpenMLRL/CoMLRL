class RewardProcessors:
    """Collection of reward processing functions to modify raw rewards."""

    @staticmethod
    def identity():
        """Return an identity processor that returns the reward unchanged."""
        return lambda x: x

    @staticmethod
    def clamp(min_val=-10.0, max_val=10.0):
        """Return a processor that clamps rewards to a range."""
        return lambda x: max(min_val, min(max_val, x))

    @staticmethod
    def scale(factor=1.0):
        """Return a processor that scales rewards by a factor."""
        return lambda x: x * factor

    @staticmethod
    def shift(value=0.0):
        """Return a processor that shifts rewards by a constant value."""
        return lambda x: x + value

    @staticmethod
    def sigmoid_scale():
        """Return a processor that applies sigmoid scaling to rewards."""
        import math

        return lambda x: 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def normalize_by_length(max_len=1000):
        """Return a processor that normalizes rewards by text length."""

        def processor(reward, text_length=None):
            if text_length is None:
                return reward
            norm_factor = min(1.0, text_length / max_len)
            return reward * norm_factor

        return processor

    @staticmethod
    def exponential_scale(factor=1.0):
        """Return a processor that applies exponential scaling to rewards."""
        import math

        return lambda x: (
            math.exp(factor * x) - 1 if x > 0 else -math.exp(-factor * x) + 1
        )
