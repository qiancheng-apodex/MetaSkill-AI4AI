"""Deterministic, dependency-free token estimates used by the schemas.

This is deliberately an *approximation*, not a model tokenizer.  A counted unit
is one ASCII word/number run, one CJK character, one run of other Unicode
letters, or one remaining non-whitespace character.  The rule is stable across
Builder and Target model choices, which is more important here than matching a
particular provider's tokenizer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


_APPROX_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_]+"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
    r"|[^\W\d_]+"
    r"|[^\s]",
    flags=re.UNICODE,
)


def approximate_token_count(text: str) -> int:
    """Return the repository's deterministic approximate token count.

    The function intentionally has no model-specific behavior.  Callers must
    pass a string; accepting arbitrary values would hide schema errors.
    """

    if not isinstance(text, str):
        raise TypeError("approximate_token_count expects a string")
    return sum(1 for _ in _APPROX_TOKEN_RE.finditer(text))


def approximate_token_count_many(values: Iterable[str]) -> int:
    """Count the textual values in a record, excluding JSON field syntax."""

    return sum(approximate_token_count(value) for value in values)
