"""
Configurable parsing of numeric strings from OCR / table cells.

Behavior is controlled with environment variables (defaults preserve common
financial-document conventions without scattering magic strings in callers).
"""

from __future__ import annotations

import math
import os
import re
from typing import Optional

_PAREN_PAIR = re.compile(r"^\(\s*(?P<inner>.+)\s*\)$")
_PAREN_OPEN_ONLY = re.compile(r"^\(\s*(?P<inner>.+)$")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _is_finite_float(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


def parse_loose_numeric(text: Optional[str]) -> Optional[float]:
    """
    Parse strings such as comma-separated amounts, currency symbols, percents,
    and optionally parenthetical negatives (when NUMERIC_PARENTHESES_AS_NEGATIVE
    is enabled).

    Environment variables
    ---------------------
    NUMERIC_PARENTHESES_AS_NEGATIVE : default true — ``(123)`` / ``( 123 )`` → -123.0
    NUMERIC_PAREN_UNCLOSED_OCR      : default true — tolerate missing closing ``)``
    """
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    t = t.replace(",", "").replace("$", "").replace("%", "").strip()
    if not t:
        return None

    parens_negative = _env_bool("NUMERIC_PARENTHESES_AS_NEGATIVE", True)
    allow_unclosed = _env_bool("NUMERIC_PAREN_UNCLOSED_OCR", True)

    if parens_negative:
        m = _PAREN_PAIR.match(t)
        if m:
            inner = m.group("inner").strip()
            if inner:
                try:
                    v = -float(inner)
                    return v if _is_finite_float(v) else None
                except (ValueError, TypeError):
                    pass
        if allow_unclosed:
            m = _PAREN_OPEN_ONLY.match(t)
            if m:
                inner = m.group("inner").strip()
                if inner:
                    try:
                        v = -float(inner)
                        return v if _is_finite_float(v) else None
                    except (ValueError, TypeError):
                        pass

    try:
        v = float(t)
        return v if _is_finite_float(v) else None
    except (ValueError, TypeError):
        return None
