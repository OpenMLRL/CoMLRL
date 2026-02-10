from typing import Optional, Sequence, Union, List

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must expose pad_token_id.")
    return tokenizer


def ensure_tokenizer(
    model: Optional[Union[str, PreTrainedModel]],
    tokenizer: Optional[PreTrainedTokenizerBase],
) -> PreTrainedTokenizerBase:
    if tokenizer is None:
        if model is None or isinstance(model, PreTrainedModel):
            raise ValueError(
                "Tokenizer must be provided when model is a PreTrainedModel instance."
            )
        tokenizer = AutoTokenizer.from_pretrained(model)
    return ensure_pad_token(tokenizer)


def _is_tokenizer_sequence(tokenizer: object) -> bool:
    return isinstance(tokenizer, Sequence) and not isinstance(
        tokenizer, (str, bytes, PreTrainedTokenizerBase)
    )


def resolve_tokenizers(
    model: Optional[Union[str, PreTrainedModel]],
    tokenizer: Optional[
        Union[PreTrainedTokenizerBase, Sequence[PreTrainedTokenizerBase]]
    ],
    agents: Optional[Sequence[object]],
) -> Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]:
    if agents is None:
        if tokenizer is not None and not _is_tokenizer_sequence(tokenizer):
            return ensure_pad_token(tokenizer)  # allow custom tokenizers
        return ensure_tokenizer(model, None)
    if tokenizer is None:
        if (
            isinstance(agents, Sequence)
            and not isinstance(agents, (str, bytes))
            and all(isinstance(src, str) for src in agents)
        ):
            tokenizers = [
                ensure_pad_token(AutoTokenizer.from_pretrained(name)) for name in agents
            ]
            return tokenizers
        raise ValueError("Tokenizer(s) must be provided when agents are model objects.")
    if _is_tokenizer_sequence(tokenizer):
        tokenizers = list(tokenizer)
        if len(tokenizers) != len(agents):
            raise ValueError("tokenizers length must match agents.")
        return [ensure_pad_token(tok) for tok in tokenizers]
    return ensure_pad_token(tokenizer)  # shared tokenizer for all agents


def resolve_tokenizer(
    model: Optional[Union[str, PreTrainedModel]],
    tokenizer: Optional[PreTrainedTokenizerBase],
    agents: Optional[Sequence[object]],
) -> PreTrainedTokenizerBase:
    tokenizers = resolve_tokenizers(model, tokenizer, agents)
    if isinstance(tokenizers, list):
        return ensure_pad_token(tokenizers[0])
    return tokenizers


def apply_tokenizer_specials(
    tokenizer: PreTrainedTokenizerBase,
    models: Sequence[object],
) -> None:
    tokenizer = ensure_pad_token(tokenizer)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id or pad_id
    for model in models:
        if model is None:
            continue
        if hasattr(model, "model") and hasattr(model.model, "config"):
            model.model.config.pad_token_id = pad_id
            model.model.config.eos_token_id = eos_id
        elif hasattr(model, "config"):
            model.config.pad_token_id = pad_id
            model.config.eos_token_id = eos_id
