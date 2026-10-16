"""Shared fixtures; every test runs offline on CPU with tiny local models."""

import pytest

from helpers import restore_rng_state, rng_state, stub_tokenizer, word_tokenizer


@pytest.fixture(autouse=True)
def _restore_global_rng():
    """Keep one test's seeding from leaking into the next."""
    state = rng_state()
    yield
    restore_rng_state(state)


@pytest.fixture(name="tokenizer")
def tokenizer_fixture():
    """A whitespace word-level tokenizer covering the joint-response vocabulary."""
    return word_tokenizer()


@pytest.fixture(name="stub_tokenizer")
def stub_tokenizer_fixture():
    """Pad/EOS attributes only, for trainers that never tokenize."""
    return stub_tokenizer()
