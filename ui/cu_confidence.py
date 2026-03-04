import re
from math import exp


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def _page_chunks_from_markdown(md: str, page_count: int) -> list[str]:
    md = (md or "").strip()
    marker = "<!-- PageBreak -->"

    if marker in md:
        chunks = [c.strip() for c in md.split(marker)]
        chunks = [c for c in chunks if c]
        if not chunks:
            return [""] * page_count
        if len(chunks) < page_count:
            chunks = chunks + ([""] * (page_count - len(chunks)))
        return chunks[:page_count]

    if not md:
        return [""] * page_count

    step = max(1, len(md) // page_count)
    chunks = [md[i : i + step].strip() for i in range(0, len(md), step)]
    if len(chunks) < page_count:
        chunks = chunks + ([""] * (page_count - len(chunks)))
    return chunks[:page_count]


def _ocr_confidence_by_page(ir: dict) -> tuple[dict, float | None]:
    """
    Returns:
      (per_page_avg_conf, overall_avg_conf)

    Expects ir["words"] as a list of:
      { "text": str, "confidence": float, "bounding_regions": [ { "page": int, "polygon": ... } ] }
    If not present, returns ({}, None).
    """
    words = ir.get("words") or []
    if not words:
        return {}, None

    per_page = {}
    overall_vals = []

    for w in words:
        conf = w.get("confidence", None)
        if conf is None:
            continue

        try:
            conf_f = float(conf)
        except Exception:
            continue

        overall_vals.append(conf_f)

        for br in (w.get("bounding_regions") or []):
            pn = br.get("page", None) or br.get("page_number", None)
            if pn is None:
                continue
            per_page.setdefault(int(pn), []).append(conf_f)

    per_page_avg = {pn: (sum(v) / len(v)) for pn, v in per_page.items() if v}
    overall_avg = (sum(overall_vals) / len(overall_vals)) if overall_vals else None
    return per_page_avg, overall_avg


def content_understanding_confidence(ir: dict) -> dict:
    """
    Deterministic confidence score computed ONLY from Content Understanding output.

    Works with:
      ir["markdown"], ir["pages"], ir["tables"]
    Optionally uses:
      ir["words"] with word level OCR confidence, if you capture it in your IR

    Returns:
      {
        "confidence_score": float (0..1),
        "confidence_level": "low" | "medium" | "high",
        "components": {...}
      }
    """
    md = (ir.get("markdown") or "").strip()
    pages = ir.get("pages") or []
    tables = ir.get("tables") or []

    page_count = max(1, len(pages))
    page_chunks = _page_chunks_from_markdown(md, page_count)

    # Text coverage and empty pages
    chars_per_page = [len(c) for c in page_chunks[:page_count]]
    avg_chars = sum(chars_per_page) / page_count
    empty_pages = sum(1 for c in chars_per_page if c < 200)
    empty_ratio = empty_pages / page_count

    # Text score: 300 chars per page weak, 1500 strong
    text_score = _clamp01(_sigmoid((avg_chars - 800) / 350))
    empty_score = _clamp01(1.0 - empty_ratio)

    # Noise score: punctuation and symbol heavy text often indicates OCR issues
    if md:
        nonword = re.findall(r"[^\w\s]", md)
        garbage_ratio = len(nonword) / max(1, len(md))
    else:
        garbage_ratio = 1.0
    noise_score = _clamp01(1.0 - garbage_ratio * 6.0)

    # Table score
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

    # OCR clarity score from word level confidences, if available
    ocr_by_page, ocr_avg = _ocr_confidence_by_page(ir)
    if ocr_avg is None:
        ocr_score = None
    else:
        # Typical OCR confidence is often around 0.6 to 0.95.
        # This maps 0.70 to low, 0.85 to strong.
        ocr_score = _clamp01(_sigmoid((ocr_avg - 0.80) / 0.06))

    # Per page clarity map
    # If OCR exists use it per page, else derive from chunk density and noise
    page_clarity = {}
    if ocr_by_page:
        for i in range(page_count):
            pn = pages[i].get("page_number", i + 1)
            if pn in ocr_by_page:
                page_clarity[int(pn)] = round(_clamp01(ocr_by_page[pn]), 3)
            else:
                page_clarity[int(pn)] = 0.55
    else:
        for i in range(page_count):
            pn = pages[i].get("page_number", i + 1)
            txt = page_chunks[i] if i < len(page_chunks) else ""
            chars = len(txt)

            if chars < 100:
                page_clarity[int(pn)] = 0.25
                continue

            nonword_p = re.findall(r"[^\w\s]", txt)
            garbage_ratio_p = len(nonword_p) / max(1, len(txt))

            density_score = _clamp01(chars / 1400)
            noise_score_p = _clamp01(1.0 - garbage_ratio_p * 6.0)

            page_clarity[int(pn)] = round(_clamp01(0.55 * density_score + 0.45 * noise_score_p), 3)

    avg_page_clarity = sum(page_clarity.values()) / max(1, len(page_clarity))

    # Final weighted score
    # If OCR is present, give it meaningful weight since it reflects scan and handwriting readability.
    if ocr_score is None:
        score = (
            0.42 * text_score +
            0.20 * table_score +
            0.20 * noise_score +
            0.18 * empty_score
        )
    else:
        score = (
            0.30 * text_score +
            0.18 * table_score +
            0.15 * noise_score +
            0.12 * empty_score +
            0.25 * ocr_score
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
            "ocr_avg_confidence": round(ocr_avg, 4) if ocr_avg is not None else None,
            "ocr_score": round(ocr_score, 3) if ocr_score is not None else None,
            "avg_page_clarity": round(avg_page_clarity, 3),
            "page_clarity_map": page_clarity,
        },
    }