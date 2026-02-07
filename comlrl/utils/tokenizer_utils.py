from __future__ import annotations

from typing import Optional, Union

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
