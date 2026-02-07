import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

Formatter = Callable[[Dict[str, Any]], str]


def build_formatters(
    formatters: Optional[Union[Formatter, Sequence[Formatter]]],
    num_agents: int,
    *,
    pass_none_external_prompts: bool = True,
) -> List[Callable[[Dict[str, Any]], str]]:
    def default_formatter(item: Dict[str, Any], external_prompts=None) -> str:
        if external_prompts is not None:
            return external_prompts
        return item.get("prompt", "")

    def wrap_formatter(fmt: Formatter) -> Formatter:
        try:
            sig = inspect.signature(fmt)
            if "external_prompts" in sig.parameters:
                if pass_none_external_prompts:
                    return lambda x, external_prompts=None, f=fmt: f(
                        x, external_prompts=external_prompts
                    )
                return lambda x, external_prompts=None, f=fmt: (
                    f(x, external_prompts=external_prompts)
                    if external_prompts is not None
                    else f(x)
                )
        except (TypeError, ValueError):
            pass
        return lambda x, external_prompts=None, f=fmt: f(x)

    if formatters is None:
        return [default_formatter for _ in range(num_agents)]
    if callable(formatters):
        return [wrap_formatter(formatters) for _ in range(num_agents)]
    if isinstance(formatters, Sequence) and not isinstance(formatters, (str, bytes)):
        if len(formatters) != num_agents:
            raise ValueError(
                "Number of formatters must match num_agents when providing a sequence."
            )
        return [wrap_formatter(f) for f in list(formatters)]
    raise ValueError("formatters must be None, a callable, or a sequence of callables.")
