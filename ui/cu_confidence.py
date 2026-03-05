import re
from math import exp


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    """
    p in [0, 100]
    """
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])

    n = len(sorted_vals)
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(sorted_vals[f])
    d = k - f
    return float(sorted_vals[f] * (1.0 - d) + sorted_vals[c] * d)


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


def _get_span(obj) -> tuple[int, int] | None:
    if not isinstance(obj, dict):
        return None

    span = obj.get("span")
    if isinstance(span, dict) and "offset" in span and "length" in span:
        try:
            start = int(span["offset"])
            end = start + int(span["length"])
            return start, end
        except Exception:
            return None

    if "offset" in obj and "length" in obj:
        try:
            start = int(obj["offset"])
            end = start + int(obj["length"])
            return start, end
        except Exception:
            return None

    return None


def _get_spans(obj) -> list[tuple[int, int]]:
    if not isinstance(obj, dict):
        return []

    spans = obj.get("spans")
    if not spans:
        one = _get_span(obj)
        return [one] if one else []

    out: list[tuple[int, int]] = []
    for s in spans:
        if isinstance(s, dict):
            se = _get_span(s)
            if se:
                out.append(se)
    return out


def _extract_words_any_shape(ir: dict) -> list[dict]:
    """
    Supports both:
      - flattened IR: ir["words"]
      - raw CU shape: ir["contents"][0]["words"] or ir["contents"][0].words
    """
    words = ir.get("words")
    if isinstance(words, list) and words:
        return words

    contents = ir.get("contents")
    if isinstance(contents, list) and contents:
        c0 = contents[0]
        if isinstance(c0, dict):
            w = c0.get("words")
            if isinstance(w, list):
                return w
        else:
            w = getattr(c0, "words", None)
            if isinstance(w, list):
                return w

    return []


def _extract_pages_any_shape(ir: dict) -> list[dict]:
    pages = ir.get("pages")
    if isinstance(pages, list) and pages:
        return pages

    contents = ir.get("contents")
    if isinstance(contents, list) and contents:
        c0 = contents[0]
        if isinstance(c0, dict):
            p = c0.get("pages")
            if isinstance(p, list):
                return p
        else:
            p = getattr(c0, "pages", None)
            if isinstance(p, list):
                return p

    return []


def _extract_tables_any_shape(ir: dict) -> list[dict]:
    tables = ir.get("tables")
    if isinstance(tables, list) and tables:
        return tables

    contents = ir.get("contents")
    if isinstance(contents, list) and contents:
        c0 = contents[0]
        if isinstance(c0, dict):
            t = c0.get("tables")
            if isinstance(t, list):
                return t
        else:
            t = getattr(c0, "tables", None)
            if isinstance(t, list):
                return t

    return []


def _ocr_confidence_by_page(ir: dict) -> tuple[dict, float | None, list[float]]:
    """
    Returns:
      (per_page_avg_conf, overall_avg_conf, all_conf_vals_raw)

    Looks for word confidence and page info via bounding regions.
    """
    words = _extract_words_any_shape(ir)
    if not words:
        return {}, None, []

    per_page: dict[int, list[float]] = {}
    overall_vals: list[float] = []

    for w in words:
        if not isinstance(w, dict):
            # if SDK objects slip in, do best effort
            conf = getattr(w, "confidence", None)
            brs = getattr(w, "bounding_regions", None) or []
        else:
            conf = w.get("confidence", None)
            brs = (w.get("bounding_regions") or [])

        if conf is None:
            continue

        try:
            conf_f = float(conf)
        except Exception:
            continue

        overall_vals.append(conf_f)

        for br in brs:
            if isinstance(br, dict):
                pn = br.get("page", None) or br.get("page_number", None)
            else:
                pn = getattr(br, "page", None) or getattr(br, "page_number", None) or getattr(br, "page_number", None)

            if pn is None:
                continue
            per_page.setdefault(int(pn), []).append(conf_f)

    per_page_avg = {pn: (sum(v) / len(v)) for pn, v in per_page.items() if v}
    overall_avg = (sum(overall_vals) / len(overall_vals)) if overall_vals else None
    return per_page_avg, overall_avg, overall_vals


def _table_cell_ocr_confidence(ir: dict) -> tuple[float | None, dict]:
    """
    Computes avg OCR confidence for table cells using span overlap.

    Requires spans to exist in words and cells.
    If spans are missing, this returns (None, {}).
    """
    tables = _extract_tables_any_shape(ir)
    words = _extract_words_any_shape(ir)
    if not tables or not words:
        return None, {}

    word_spans: list[tuple[int, int, float]] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        se = _get_span(w)
        if not se:
            continue
        conf = w.get("confidence", None)
        if conf is None:
            continue
        try:
            conf_f = float(conf)
        except Exception:
            continue
        word_spans.append((se[0], se[1], conf_f))

    if not word_spans:
        return None, {}

    per_table_avg: dict[int, float] = {}
    all_cell_avgs: list[float] = []

    for ti, t in enumerate(tables, start=1):
        if not isinstance(t, dict):
            continue

        cell_avgs: list[float] = []
        for c in (t.get("cells") or []):
            if not isinstance(c, dict):
                continue
            spans = _get_spans(c)
            if not spans:
                continue

            cell_word_confs: list[float] = []
            for (span_start, span_end) in spans:
                for (w_start, w_end, w_conf) in word_spans:
                    if w_start >= span_start and w_end <= span_end:
                        cell_word_confs.append(w_conf)

            if not cell_word_confs:
                continue

            cell_avg = sum(cell_word_confs) / len(cell_word_confs)
            cell_avgs.append(cell_avg)
            all_cell_avgs.append(cell_avg)

        if cell_avgs:
            per_table_avg[ti] = sum(cell_avgs) / len(cell_avgs)

    overall = (sum(all_cell_avgs) / len(all_cell_avgs)) if all_cell_avgs else None
    return overall, {k: round(v, 4) for k, v in per_table_avg.items()}


def content_understanding_confidence(ir: dict) -> dict:
    """
    Deterministic confidence score computed from Content Understanding output.

    Adds portal style OCR metrics:
      - portal_word_ocr_avg_confidence
      - portal_word_ocr_p50_confidence
      - portal_word_ocr_p90_confidence
      - portal_word_ocr_by_page_avg
    """
    md = (ir.get("markdown") or "").strip()
    pages = _extract_pages_any_shape(ir)
    tables = _extract_tables_any_shape(ir)

    page_count = max(1, len(pages)) if pages else 1
    page_chunks = _page_chunks_from_markdown(md, page_count)

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
        if not isinstance(t, dict):
            continue
        for c in (t.get("cells") or []):
            if not isinstance(c, dict):
                continue
            cell_count += 1
            txt = (c.get("text") or c.get("content") or "").strip()
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

    ocr_by_page, ocr_avg, ocr_vals = _ocr_confidence_by_page(ir)

    # Portal style values (raw)
    ocr_vals_sorted = sorted(ocr_vals)
    portal_ocr_avg = round(ocr_avg, 4) if ocr_avg is not None else None
    portal_ocr_p50 = round(_percentile(ocr_vals_sorted, 50) or 0.0, 4) if ocr_vals_sorted else None
    portal_ocr_p90 = round(_percentile(ocr_vals_sorted, 90) or 0.0, 4) if ocr_vals_sorted else None
    portal_ocr_by_page = {int(k): round(v, 4) for k, v in (ocr_by_page or {}).items()} or None

    # Scaled OCR score used only for the blended "confidence_score"
    if ocr_avg is None:
        ocr_score = None
    else:
        ocr_score = _clamp01(_sigmoid((ocr_avg - 0.80) / 0.06))

    table_ocr_avg, table_ocr_per_table = _table_cell_ocr_confidence(ir)
    if table_ocr_avg is None:
        table_ocr_score = None
    else:
        table_ocr_score = _clamp01(_sigmoid((table_ocr_avg - 0.80) / 0.06))
        table_score = _clamp01(0.75 * table_score + 0.25 * table_ocr_score)

    # Page clarity map stays as before
    page_clarity: dict[int, float] = {}
    if pages:
        if ocr_by_page:
            for i in range(page_count):
                pn = (pages[i].get("page_number", i + 1) if isinstance(pages[i], dict) else i + 1)
                if pn in ocr_by_page:
                    page_clarity[int(pn)] = round(_clamp01(ocr_by_page[pn]), 3)
                else:
                    page_clarity[int(pn)] = 0.55
        else:
            for i in range(page_count):
                pn = (pages[i].get("page_number", i + 1) if isinstance(pages[i], dict) else i + 1)
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
    else:
        page_clarity = {1: 0.55}

    avg_page_clarity = sum(page_clarity.values()) / max(1, len(page_clarity))

    # Final blended score (not portal, your deterministic score)
    if ocr_score is None:
        score = 0.42 * text_score + 0.20 * table_score + 0.20 * noise_score + 0.18 * empty_score
    else:
        score = 0.30 * text_score + 0.18 * table_score + 0.15 * noise_score + 0.12 * empty_score + 0.25 * ocr_score

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
        "portal_word_ocr_avg_confidence": portal_ocr_avg,
        "portal_word_ocr_p50_confidence": portal_ocr_p50,
        "portal_word_ocr_p90_confidence": portal_ocr_p90,
        "portal_word_ocr_by_page_avg": portal_ocr_by_page,
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
            "table_cell_ocr_avg_confidence": round(table_ocr_avg, 4) if table_ocr_avg is not None else None,
            "table_cell_ocr_score": round(table_ocr_score, 3) if table_ocr_score is not None else None,
            "table_cell_ocr_map": table_ocr_per_table or None,
            "garbage_ratio": round(garbage_ratio, 4),
            "noise_score": round(noise_score, 3),
            "ocr_avg_confidence": round(ocr_avg, 4) if ocr_avg is not None else None,
            "ocr_score": round(ocr_score, 3) if ocr_score is not None else None,
            "avg_page_clarity": round(avg_page_clarity, 3),
            "page_clarity_map": page_clarity,
        },
    }