from typing import Any


def unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)
