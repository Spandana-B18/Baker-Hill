"""
financial_spreading/statement_mapper.py

Maps extracted financial rows to the Chart of Accounts schema using
a precision first strategy:

1. Exact and keyword style match
2. Weighted fuzzy similarity
3. Basic normalization for noisy tax form labels
4. Keeps all matched rows, including page 1 form fields and Schedule L rows
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple


_STRIP_PUNCT = re.compile(r"[^a-z0-9 ]")
_STOPWORDS: Set[str] = {
    "and",
    "or",
    "the",
    "of",
    "in",
    "on",
    "to",
    "for",
    "from",
    "at",
    "by",
    "with",
    "net",
    "total",
    "current",
    "other",
}


def _normalize(text: str) -> str:
    """
    Lower case, strip punctuation, collapse whitespace.
    """
    text = text or ""
    text = text.lower().strip()
    text = _STRIP_PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonicalize_source_label(text: str) -> str:
    """
    Normalize tax form style labels before matching.

    Examples:
      F Total assets (see instructions)   -> Total assets
      Total assets [Beginning of tax year] -> Total assets beginning of tax year
    """
    t = text or ""
    t = re.sub(r"^[A-Z]\s+", "", t)
    t = re.sub(r"\(see instructions\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\(attach [^)]+\)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"[\.\:]+", " ", t)
    t = t.replace("&", " and ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> Set[str]:
    norm = _normalize(text)
    if not norm:
        return set()
    return {tok for tok in norm.split() if tok and tok not in _STOPWORDS}


def _similarity(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _token_overlap_score(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return inter / union


def _keyword_exact_match(
    label: str,
    schema: List[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], float, str]]:
    """
    Check whether normalized label contains any schema keyword or vice versa.
    Returns best (item, score, matched_candidate) or None.
    """
    norm_label = _normalize(label)

    best_item: Optional[Dict[str, Any]] = None
    best_score = 0.0
    best_candidate = ""

    for item in schema:
        candidates = [item["row_label"]] + item.get("keywords", [])
        for candidate in candidates:
            norm_cand = _normalize(candidate)
            if not norm_cand:
                continue

            if norm_cand == norm_label:
                score = 0.95
            elif norm_cand in norm_label or norm_label in norm_cand:
                score = 0.90
            else:
                continue

            if score > best_score:
                best_score = score
                best_item = item
                best_candidate = candidate

    if best_item is None:
        return None

    return best_item, best_score, best_candidate


def match_row(
    row_label: str,
    schema: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """
    Return (best_schema_item, score, matched_candidate) for row_label.
    """
    cleaned_label = _canonicalize_source_label(row_label)

    exact = _keyword_exact_match(cleaned_label, schema)
    if exact:
        item, score, candidate = exact
        return item, score, candidate

    best_match: Optional[Dict[str, Any]] = None
    best_score = 0.0
    best_candidate = ""

    for item in schema:
        candidates = [item["row_label"]] + item.get("keywords", [])
        for candidate in candidates:
            seq_score = _similarity(cleaned_label, candidate)
            overlap_score = _token_overlap_score(cleaned_label, candidate)

            score = (0.65 * seq_score) + (0.35 * overlap_score)

            if score > best_score:
                best_score = score
                best_match = item
                best_candidate = candidate

    return best_match, best_score, best_candidate


def _build_reasoning(
    source_label: str,
    matched_candidate: str,
    match: Dict[str, Any],
    score: float,
    page: Optional[int],
    year: Optional[int],
) -> str:
    page_str = f" on page {page}" if page else ""
    year_str = f" for year {year}" if year else ""

    if score >= 0.90:
        method = "exact keyword match"
    elif score >= 0.75:
        method = "high confidence fuzzy match"
    else:
        method = "fuzzy similarity match"

    return (
        f"Matched '{source_label}'{page_str}{year_str} to "
        f"'{match['row_label']}' ({match['chart_of_account_line']}) "
        f"using candidate '{matched_candidate}' via {method} "
        f"(score: {score:.2f})."
    )


def spread_statement(
    rows: List[Dict[str, Any]],
    schema: List[Dict[str, Any]],
    threshold: float = 0.65,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Map a list of extracted rows to the COA schema.

    Output row contract:
        chart_of_account_line, row_label, year, value,
        confidence, reference, source_label, reasoning
    """
    mapped_rows: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []

    for row in rows:
        source_label = row.get("label", "")
        match, score, matched_candidate = match_row(source_label, schema)

        if match is None or score < threshold:
            unmatched.append(
                {
                    "source_label": source_label,
                    "values": row.get("values", []),
                    "page": row.get("page"),
                    "match_score": round(score, 3),
                }
            )
            continue

        values_list = row.get("values", [])
        has_period_values = any(
            (v.get("period") if isinstance(v, dict) else None) is not None
            for v in values_list
        )
        for val_entry in values_list:
            value = val_entry["value"] if isinstance(val_entry, dict) else val_entry
            original_value = val_entry.get("original_value", "") if isinstance(val_entry, dict) else ""
            year = val_entry.get("year") if isinstance(val_entry, dict) else None
            period = val_entry.get("period") if isinstance(val_entry, dict) else None
            if period is None and has_period_values:
                continue

            reasoning = _build_reasoning(
                source_label=source_label,
                matched_candidate=matched_candidate,
                match=match,
                score=score,
                page=row.get("page"),
                year=year,
            )

            output_row_label = match["row_label"]
            if period:
                output_row_label = f"{output_row_label} [{period}]"

            mapped_rows.append(
                {
                    "chart_of_account_line": match["chart_of_account_line"],
                    "row_label": output_row_label,
                    "year": year,
                    "value": value,
                    "original_value": original_value,
                    "confidence": round(score, 3),
                    "reference": row.get("page"),
                    "source_label": source_label,
                    "reasoning": reasoning,
                }
            )

    return mapped_rows, unmatched
