from typing import Optional, Sequence, Union

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


def resolve_tokenizer(
    model: Optional[Union[str, PreTrainedModel]],
    tokenizer: Optional[PreTrainedTokenizerBase],
    agents: Optional[Sequence[object]],
) -> PreTrainedTokenizerBase:
    if agents is not None and tokenizer is None:
        raise ValueError("Tokenizer must be provided when agents are passed.")
    if agents is None:
        return ensure_tokenizer(model, tokenizer)
    return ensure_pad_token(tokenizer)


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
