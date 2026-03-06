import re
from math import exp
from typing import Any, Dict, List, Tuple, Union


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def content_understanding_confidence(ir: dict) -> dict:
    """
    Deterministic document-level confidence score computed ONLY from
    Content Understanding output (markdown + pages + tables).
    """
    md = (ir.get("markdown") or "").strip()
    pages = ir.get("pages") or []
    tables = ir.get("tables") or []

    page_count = max(1, len(pages))

    marker = "<!-- PageBreak -->"
    if marker in md:
        page_chunks = [c.strip() for c in md.split(marker) if c.strip()]
    else:
        step = max(1, len(md) // page_count) if md else 1
        page_chunks = [md[i : i + step].strip() for i in range(0, len(md), step)]
        page_chunks = page_chunks[:page_count] if page_chunks else [""]

    chars_per_page = [len(c) for c in page_chunks[:page_count]]
    avg_chars = sum(chars_per_page) / page_count
    empty_pages = sum(1 for c in chars_per_page if c < 200)
    empty_ratio = empty_pages / page_count

    text_score = _clamp01(_sigmoid((avg_chars - 800) / 350))
    empty_score = _clamp01(1.0 - empty_ratio)

    if md:
        nonword = re.findall(r"[^\w\s]", md)
        garbage_ratio = len(nonword) / max(1, len(md))
    else:
        garbage_ratio = 1.0
    noise_score = _clamp01(1.0 - garbage_ratio * 6.0)

    table_count = len(tables)
    cell_count = 0
    filled_cells = 0
    table_text_chars = 0

    for t in tables:
        for c in (t.get("cells") or []):
            cell_count += 1
            txt = (c.get("text") or "").strip()
            if txt:
                filled_cells += 1
                table_text_chars += len(txt)

    if table_count == 0:
        table_score = 0.6
        filled_ratio = None
    else:
        filled_ratio = filled_cells / max(1, cell_count)
        size_bonus = _clamp01(table_text_chars / 4000)
        table_score = _clamp01(0.55 + 0.35 * filled_ratio + 0.10 * size_bonus)

    score = (
        0.45 * text_score +
        0.20 * table_score +
        0.20 * noise_score +
        0.15 * empty_score
    )
    score = round(_clamp01(score), 3)

    if score < 0.55:
        level = "low"
    elif score < 0.75:
        level = "medium"
    else:
        level = "high"

    return {
        "confidence_score": score,
        "confidence_level": level,
        "components": {
            "page_count": page_count,
            "avg_chars_per_page": round(avg_chars, 1),
            "empty_page_ratio": round(empty_ratio, 3),
            "text_score": round(text_score, 3),
            "table_count": table_count,
            "cell_count": cell_count,
            "filled_cell_ratio": round(filled_ratio, 3) if filled_ratio is not None else None,
            "table_text_chars": table_text_chars,
            "table_score": round(table_score, 3),
            "garbage_ratio": round(garbage_ratio, 4),
            "noise_score": round(noise_score, 3),
        },
    }


def _flatten_text_sources(ir: dict) -> str:
    md = (ir.get("markdown") or "").strip()
    tables = ir.get("tables") or []
    table_texts: List[str] = []
    for t in tables:
        for c in (t.get("cells") or []):
            txt = (c.get("text") or "").strip()
            if txt:
                table_texts.append(txt)
    blob = md + "\n" + "\n".join(table_texts)
    return blob


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_number(s: str) -> bool:
    return bool(re.fullmatch(r"[\$]?\s*[-+]?\d[\d,]*([.]\d+)?", s.strip()))


def _looks_like_dateish(s: str) -> bool:
    # very loose: catches 2024, 01/31/2024, Jan 2024 etc
    s2 = s.lower()
    if re.search(r"\b(19|20)\d{2}\b", s2):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", s2):
        return True
    if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", s2):
        return True
    return False


def _value_in_text(value: Any, text_blob: str) -> float:
    """
    Returns 0..1 score for how well value can be found in CU text.
    """
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        v = str(value)
        return 1.0 if v in text_blob else 0.4

    if isinstance(value, bool):
        v = "true" if value else "false"
        return 1.0 if v in text_blob else 0.4

    if isinstance(value, str):
        v = value.strip()
        if not v:
            return 0.0

        vn = _norm(v)
        if len(vn) < 3:
            return 0.3

        if _looks_like_number(v):
            digits = re.sub(r"[^\d.]", "", v)
            if digits and digits in text_blob:
                return 1.0
            return 0.45

        if _looks_like_dateish(v):
            year = re.search(r"\b(19|20)\d{2}\b", vn)
            if year and year.group(0) in text_blob:
                return 0.8
            return 0.5

        if vn in text_blob:
            return 1.0

        # partial match fallback for long strings
        if len(vn) >= 10:
            head = vn[:10]
            if head in text_blob:
                return 0.7

        return 0.35

    # lists and dicts are handled by walking leaves
    return 0.0


def _is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _walk_leaves(obj: Any, path: Tuple[Union[str, int], ...] = ()) -> List[Tuple[Tuple[Union[str, int], ...], Any]]:
    out: List[Tuple[Tuple[Union[str, int], ...], Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk_leaves(v, path + (k,)))
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_leaves(v, path + (i,)))
        return out
    out.append((path, obj))
    return out


def _get_by_path(obj: Any, path: Tuple[Union[str, int], ...]) -> Any:
    cur = obj
    for p in path:
        if isinstance(cur, dict) and isinstance(p, str) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and isinstance(p, int) and 0 <= p < len(cur):
            cur = cur[p]
        else:
            return None
    return cur


def content_understanding_field_confidence(
    ir: dict,
    payload: dict,
    evidence: dict | None = None,
    base_doc_score: float | None = None,
) -> dict:
    """
    Compute per-field confidence scores for payload leaf paths.

    Output:
      {
        "field_confidence": { "payload.a.b": 0.83, "payload.items.0.amount": 0.71, ... },
        "summary": { "avg": 0.77, "min": 0.32, "max": 0.98, "fields": 42 }
      }
    """
    if base_doc_score is None:
        base_doc_score = (content_understanding_confidence(ir).get("confidence_score") or 0.0)

    text_blob = _norm(_flatten_text_sources(ir))

    evidence = evidence or {}

    scores: Dict[str, float] = {}
    leafs = _walk_leaves(payload, ())

    for rel_path, value in leafs:
        key = "payload." + ".".join(str(p) for p in rel_path)

        if _is_empty_value(value):
            scores[key] = 0.0
            continue

        text_hit = _value_in_text(value, text_blob)

        ev_val = _get_by_path(evidence, rel_path)
        ev_present = 0.0 if _is_empty_value(ev_val) else 1.0

        score = (
            0.45 * float(base_doc_score) +
            0.40 * float(text_hit) +
            0.15 * float(ev_present)
        )
        scores[key] = round(_clamp01(score), 3)

    vals = list(scores.values())
    if vals:
        summary = {
            "avg": round(sum(vals) / len(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "fields": len(vals),
        }
    else:
        summary = {"avg": 0.0, "min": 0.0, "max": 0.0, "fields": 0}

    return {"field_confidence": scores, "summary": summary}