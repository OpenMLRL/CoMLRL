class RewardProcessors:
    """Collection of reward processing functions to modify raw rewards."""

    @staticmethod
    def scale(factor=1.0):
        """Return a processor that scales rewards by a factor."""
        return lambda x: x * factor

    @staticmethod
    def shift(value=0.0):
        """Return a processor that shifts rewards by a constant value."""
        return lambda x: x + value
