"""
financial_spreading/extract_rows.py
 
Converts the raw Azure Content Understanding JSON into a flat list of
financial rows suitable for COA mapping.
 
Key fixes in this version:
1. Uses structured table cells first
2. Also runs line fallback even when tables exist
3. Preserves period context for multi-value table rows
   e.g. "Total assets [Beginning of tax year]" vs "Total assets [End of tax year]"
4. Avoids duplicate fallback rows when the same label/value/page already came from tables
5. Captures standalone tax form fields like page 1 "F Total assets"
"""
 
from __future__ import annotations
 
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from core.conf_score import ContentUnderstandingClient
from core.numeric_parse import parse_loose_numeric
 
# Make sure project root is on sys.path so we can import core
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
 
 
 
# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
 
# Matches 4-digit years 1990–2029
_YEAR_RE = re.compile(r"\b(19[9][0-9]|20[0-2][0-9])\b")
 
 
def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default
 
 
_parse_numeric = parse_loose_numeric


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())
 
 
def _normalize_period(text: str) -> str:
    t = _normalize_label(text)
    t = t.replace("(b)", "").replace("(d)", "").strip()
    return t
 
 
# Tax-form line-reference pattern: "1", "2a", "15c", "21" etc.
_LINE_REF_RE = re.compile(r"^\d{1,3}[a-zA-Z]?$")
 
 
def _is_line_reference_cell(text: str) -> bool:
    """
    Return True if the cell text looks like a tax-form line reference
    rather than a real descriptive label (e.g. '1', '2a', '15', '21c').
    These should never be treated as labels or as financial values.
    """
    return bool(_LINE_REF_RE.match(text.strip()))
 
 
# def _is_financial_amount(original_text: str, parsed_value: float) -> bool:
#     """
#     Return True only when the text is a genuine financial dollar amount.
 
#     A value is treated as financial when it has at least one of:
#       - comma-thousands formatting  e.g. "1,234,567"
#       - parentheses negatives        e.g. "(500)"
#       - explicit dollar sign         e.g. "$500"
#       - decimal cents                e.g. "500.00" or "1,234.56"
#       - magnitude >= 1000            e.g. 1000 or -5000
 
#     Small plain integers (1–999) without any of the above markers are
#     treated as reference numbers / line counters and rejected.
#     """
#     t = original_text.strip()
#     if "," in t:
#         return True
#     if re.match(r"^\([\d.,]+\)$", t):
#         return True
#     if "$" in t:
#         return True
#     if "." in t:
#         return True
#     if abs(parsed_value) >= 1000:
#         return True
#     return False
 
 
def _is_financial_amount(original_text: str, parsed_value: float) -> bool:
    """
    Determine whether a parsed numeric value should be retained as a financial candidate.
 
    Design principles:
    ------------------------------------------------
    - DO NOT filter based on magnitude (e.g., >= 1000) → avoids losing valid values
    - DO NOT rely on formatting ($, commas, decimals) → tax docs often lack these
    - Keep extraction stage lossless
    - Downstream logic (label pairing + COA mapping) will handle relevance
 
    Parameters:
    ------------------------------------------------
    original_text : str
        Raw OCR text corresponding to the numeric value
 
    parsed_value : float
        Parsed numeric value
 
    Returns:
    ------------------------------------------------
    bool
        True if numeric and valid candidate
    """
 
    # Reject if parsing failed
    if parsed_value is None:
        return False
 
    # Reject empty or whitespace-only text
    if not original_text or not original_text.strip():
        return False
 
    # Keep ALL numeric values (critical for tax documents)
    return True
 
def _is_valid_financial_label(label: str) -> bool:
    """
    Filter out non-meaningful labels such as:
    - line numbers (1, 2, 3)
    - tax line codes (1a, 2b, 3c)
    - very short or noisy tokens
 
    This ensures we do NOT treat reference markers as financial labels.
    """
 
    if not label:
        return False
 
    label = label.strip().lower()
 
    # Reject pure numbers (e.g., "1", "2")
    if label.isdigit():
        return False
 
    # Reject patterns like "1a", "2b", "3c"
    if re.match(r"^\d+[a-z]?$", label):
        return False
 
    # Reject very short labels (likely noise)
    if len(label) <= 2:
        return False
 
    return True
 
 
def _build_value_period_suffix(period_label: Optional[str]) -> str:
    if not period_label:
        return ""
    return f" [{period_label}]"


def _detect_value_columns(
    row_map: Dict[int, Dict[int, str]],
) -> Optional[set]:
    """
    Identify which column indices are genuine financial value columns.

    Strategy:
    - For each column, collect all parsed numeric values across all rows.
    - A column is classified as a VALUE column if at least 2 of its values
      are >= 100 (indicating financial magnitudes, not line numbers).
    - A column where ALL values are small integers <= 99 and appear
      sequentially is classified as a LINE NUMBER column and excluded.
    - If no confident value columns are found, returns None so the caller
      falls back to keeping all non-line-reference numbers (safe fallback).

    Returns
    -------
    set of column indices that are confirmed value columns, or None if
    detection was inconclusive (caller should not filter in that case).
    """
    from collections import defaultdict as _dd

    col_values: Dict[int, List[float]] = _dd(list)

    for col_map in row_map.values():
        for col_idx, text in col_map.items():
            if _is_line_reference_cell(text):
                continue
            parsed = _parse_numeric(text)
            if parsed is not None:
                col_values[col_idx].append(parsed)

    if not col_values:
        return None

    value_cols: set = set()

    for col_idx, vals in col_values.items():
        if not vals:
            continue

        large_count = sum(1 for v in vals if abs(v) >= 100)

        # Column is a value column if at least 2 entries are >= 100
        # OR if it has only one row but that value is >= 100
        if large_count >= 2 or (len(vals) == 1 and abs(vals[0]) >= 100):
            value_cols.add(col_idx)
            continue

        # Check if this looks like a sequential line number column
        # (all values are small integers 1-99 in ascending order)
        int_vals = [int(v) for v in vals if v == int(v) and 0 < v <= 99]
        if len(int_vals) == len(vals) and int_vals == sorted(int_vals):
            # Sequential small integers → line number column, exclude it
            continue

        # Small values but NOT sequential → could be legitimate small amounts
        # (e.g. interest income of $4) — include in value columns
        value_cols.add(col_idx)

    # If detection found no confident value columns, return None (safe fallback)
    return value_cols if value_cols else None
 
 
def _detect_column_years(
    row_map: Dict[int, Dict[int, str]],
    num_header_rows: int = 5,
) -> Dict[int, int]:
    """
    Scan the first few rows for year-like column headers.
    Returns {column_index: fiscal_year}.
    """
    col_years: Dict[int, int] = {}
    for row_idx in sorted(row_map.keys())[:num_header_rows]:
        for col_idx, text in row_map[row_idx].items():
            m = _YEAR_RE.search(text)
            if m:
                col_years[col_idx] = int(m.group(1))
    return col_years
 
 
def _detect_column_periods(
    row_map: Dict[int, Dict[int, str]],
    num_header_rows: int = 8,
) -> Dict[int, str]:
    """
    Detect semantic column periods from header rows.
 
    This is especially useful for tax forms like 1120-S Schedule L where
    the same row has multiple value columns such as:
      Beginning of tax year / End of tax year
    """
    col_periods: Dict[int, str] = {}
 
    header_rows = sorted(row_map.keys())[:num_header_rows]
 
    for row_idx in header_rows:
        for col_idx, text in row_map[row_idx].items():
            norm = _normalize_period(text)
 
            if not norm:
                continue
 
            if "beginning of tax year" in norm:
                col_periods[col_idx] = "Beginning of tax year"
            elif "end of tax year" in norm:
                col_periods[col_idx] = "End of tax year"
 
    # Secondary heuristic for forms that put (a)(b)(c)(d) under a merged header.
    # In Schedule L, numeric values are usually in (b) and (d).
    # If we already found one of them, map the sibling code too.
    # Example:
    #   (a) label col / blank
    #   (b) beginning value col
    #   (c) label col / blank
    #   (d) end value col
    for row_idx in header_rows:
        for col_idx, text in row_map[row_idx].items():
            norm = _normalize_label(text)
            if norm == "(b)" and col_idx not in col_periods:
                col_periods[col_idx] = "Beginning of tax year"
            elif norm == "(d)" and col_idx not in col_periods:
                col_periods[col_idx] = "End of tax year"
 
    return col_periods


def _detect_two_column_split(
    row_map: Dict[int, Dict[int, str]],
) -> Optional[int]:
    """
    Detect if a table uses a two-column side-by-side layout (common in
    Schedule K-1 Part III).

    In this layout each table row contains two independent items:
      col 0  col 1              col 2      col 3  col 4                       col 5
      1      Ordinary income    456,123    14     Self-employment earnings    50,000

    We detect this by finding rows that have TWO label-like cells and recording
    the column index of the second label. If that column is consistent across
    multiple rows, we return it as the split point.

    For single-column tables (Schedule L, financial statements) every row has
    exactly one label, so this returns None and the caller uses its current logic.
    """
    from collections import Counter as _Counter

    right_label_cols: List[int] = []

    for col_map in row_map.values():
        label_cols = [
            col_idx
            for col_idx in sorted(col_map.keys())
            if (
                col_map[col_idx].strip()
                and _parse_numeric(col_map[col_idx]) is None
                and not _is_line_reference_cell(col_map[col_idx])
                and len(col_map[col_idx].strip()) > 3
            )
        ]
        if len(label_cols) >= 2:
            right_label_cols.append(label_cols[1])

    if not right_label_cols:
        return None

    counter = _Counter(right_label_cols)
    most_common_col, count = counter.most_common(1)[0]
    # Require the same split column to appear in at least 2 rows
    return most_common_col if count >= 2 else None


def _rows_from_cell_group(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert a group of cells (same page + table_index) into financial rows.
    Cells come from ContentUnderstandingClient.extract_table_cells_with_confidence
    so field names are: row_index, column_index, text, page, x_center.

    Layout strategy:
      - When bounding-box X-centre is available, use **physical position** to
        pair each value with its nearest label — completely bypasses OCR
        column-index issues in two-column forms like K-1 Part III.
      - When X-centre is missing, fall back to column-index logic with
        two-column split detection.
    """
    if not cells:
        return []

    page_number = cells[0].get("page", 0)

    # Build per-row maps
    row_map: Dict[int, Dict[int, str]] = {}
    conf_map: Dict[int, Dict[int, float]] = {}
    x_map: Dict[int, Dict[int, float]] = {}     # physical X-centre per cell
    for cell in cells:
        r = _safe_int(cell.get("row_index"), 0)
        c = _safe_int(cell.get("column_index"), 0)
        row_map.setdefault(r, {})[c] = (cell.get("text") or "").strip()
        conf_map.setdefault(r, {})[c] = cell.get("confidence", 0.0) or 0.0
        xc = cell.get("x_center")
        if xc is not None:
            x_map.setdefault(r, {})[c] = float(xc)

    col_years = _detect_column_years(row_map)
    col_periods = _detect_column_periods(row_map)
    value_cols = _detect_value_columns(row_map)

    # Check if we have enough X-centre data for layout-aware pairing.
    # If most cells have x_center, use the physical-position path.
    total_cells = sum(len(cm) for cm in row_map.values())
    total_x     = sum(len(xm) for xm in x_map.values())
    use_x_layout = total_x > total_cells * 0.5

    # Precompute two-column split for Path B (column-index fallback)
    two_col_split = None if use_x_layout else _detect_two_column_split(row_map)

    rows: List[Dict[str, Any]] = []

    for row_idx in sorted(row_map.keys()):
        col_map = row_map[row_idx]
        if not col_map:
            continue

        max_col = max(col_map.keys())
        ordered = [(i, col_map.get(i, "")) for i in range(max_col + 1)]
        row_x   = x_map.get(row_idx, {})

        # ── Path A: layout-aware pairing (X-centre available) ────────────
        if use_x_layout:
            # Collect ALL labels in this row
            all_labels: List[Tuple[int, str, Optional[float]]] = []
            for col_idx, cell_text in ordered:
                if (
                    cell_text
                    and _parse_numeric(cell_text) is None
                    and not _is_line_reference_cell(cell_text)
                    and len(cell_text.strip()) > 2
                ):
                    all_labels.append((col_idx, cell_text, row_x.get(col_idx)))

            if not all_labels:
                continue

            # Collect ALL numeric values in this row
            all_values: List[Dict[str, Any]] = []
            for col_idx, cell_text in ordered:
                if _is_line_reference_cell(cell_text):
                    continue
                parsed = _parse_numeric(cell_text)
                if parsed is None:
                    continue
                if value_cols is not None and col_idx not in value_cols:
                    continue
                all_values.append({
                    "value": parsed,
                    "original_value": cell_text.strip(),
                    "year": col_years.get(col_idx),
                    "period": col_periods.get(col_idx),
                    "column_index": col_idx,
                    "confidence": conf_map.get(row_idx, {}).get(col_idx, 0.0),
                    "x_center": row_x.get(col_idx),
                })

            if not all_values:
                continue

            # If any value has a period tag, keep all values for every label
            # (Schedule L multi-period tables)
            has_periods = any(v.get("period") for v in all_values)

            if has_periods or len(all_labels) <= 1:
                # Single-label row or period-tagged: assign all values to
                # the first (or only) label
                lbl_col, lbl_text, _ = all_labels[0]
                rows.append({
                    "label": lbl_text,
                    "values": all_values,
                    "page": page_number,
                    "row_index": row_idx,
                    "label_confidence": conf_map.get(row_idx, {}).get(lbl_col, 0.0),
                })
            else:
                # Multiple labels, no period tags → assign each value to
                # its physically nearest label by X-centre distance.
                label_values: Dict[int, List[Dict[str, Any]]] = {
                    lbl_col: [] for lbl_col, _, _ in all_labels
                }

                for v in all_values:
                    v_x = v.get("x_center")
                    best_lbl_col = all_labels[0][0]
                    best_dist = float("inf")
                    for lbl_col, _, lbl_x in all_labels:
                        if v_x is not None and lbl_x is not None:
                            d = abs(v_x - lbl_x)
                        else:
                            # Fallback: column-index distance
                            d = abs(v.get("column_index", 0) - lbl_col)
                        if d < best_dist:
                            best_dist = d
                            best_lbl_col = lbl_col
                    label_values[best_lbl_col].append(v)

                for lbl_col, lbl_text, _ in all_labels:
                    assigned = label_values.get(lbl_col, [])
                    if not assigned:
                        continue
                    rows.append({
                        "label": lbl_text,
                        "values": assigned,
                        "page": page_number,
                        "row_index": row_idx,
                        "label_confidence": conf_map.get(row_idx, {}).get(lbl_col, 0.0),
                    })

            continue   # skip Path B for this row

        # ── Path B: column-index fallback (no X-centre data) ─────────────

        if two_col_split is not None:
            halves = [
                [(ci, ct) for ci, ct in ordered if ci < two_col_split],
                [(ci, ct) for ci, ct in ordered if ci >= two_col_split],
            ]
        else:
            halves = [ordered]

        for half_ordered in halves:
            label = ""
            label_col = -1
            for col_idx, cell_text in half_ordered:
                if (
                    cell_text
                    and _parse_numeric(cell_text) is None
                    and not _is_line_reference_cell(cell_text)
                ):
                    label = cell_text
                    label_col = col_idx
                    break

            if not label:
                continue

            label_conf = conf_map.get(row_idx, {}).get(label_col, 0.0)

            values: List[Dict[str, Any]] = []
            for col_idx, cell_text in half_ordered:
                if col_idx == label_col:
                    continue
                if _is_line_reference_cell(cell_text):
                    continue
                parsed = _parse_numeric(cell_text)
                if parsed is None:
                    continue
                if value_cols is not None and col_idx not in value_cols:
                    continue

                period_label = col_periods.get(col_idx)
                values.append({
                    "value": parsed,
                    "original_value": cell_text.strip(),
                    "year": col_years.get(col_idx),
                    "period": period_label,
                    "column_index": col_idx,
                    "confidence": conf_map.get(row_idx, {}).get(col_idx, 0.0),
                    "x_center": None,
                })

            if values:
                if len(values) > 1 and not any(v.get("period") for v in values):
                    values = [
                        min(values, key=lambda v: abs(
                            (v.get("column_index") or 0) - label_col
                        ))
                    ]

                rows.append({
                    "label": label,
                    "values": values,
                    "page": page_number,
                    "row_index": row_idx,
                    "label_confidence": label_conf,
                })

    return rows
 
def _rows_from_lines_fallback(
    lines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Fallback for documents like 1120-S that have key/value fields outside tables.
 
    Important behavior:
    - keeps standalone form fields like "Total assets" -> "1500000"
    - skips obvious junk
    - does not try to infer beginning/end period here
    """
    _SYMBOL_ONLY_RE = re.compile(r"^[\W_\d]+$")
 
    def _is_unusable_label(text: str) -> bool:
        if not text:
            return True
        if _SYMBOL_ONLY_RE.match(text):
            return True
        if len(text) <= 1:
            return True
        return False
 
    rows: List[Dict[str, Any]] = []
 
    detected_year: Optional[int] = None
    for line in lines[:60]:
        m = _YEAR_RE.search(line.get("text", ""))
        if m:
            detected_year = int(m.group(1))
            break
 
    i = 0
    while i < len(lines):
        text = (lines[i].get("text") or "").strip()
        page = lines[i].get("page", 0)
 
        if _parse_numeric(text) is not None or _is_unusable_label(text):
            i += 1
            continue
 
        values: List[Dict[str, Any]] = []
        j = i + 1
        while j < len(lines) and j <= i + 5:
            next_text = (lines[j].get("text") or "").strip()
            parsed = _parse_numeric(next_text)
            if parsed is not None:
                # Skip year-like values (e.g. "2023")
                if _YEAR_RE.fullmatch(next_text.strip()):
                    j += 1
                    continue
                # Skip IRS line reference numbers e.g. "3", "5", "1a"
                # These appear between label and value in 1040/1065 forms
                if _is_line_reference_cell(next_text):
                    j += 1
                    continue
                # Skip line reference numbers and non-financial small integers
                if not _is_financial_amount(next_text, parsed):
                    j += 1
                    continue
 
                values.append(
                    {
                        "value": parsed,
                        "original_value": next_text.strip(),
                        "year": detected_year,
                        "period": None,
                        "column_index": None,
                        "confidence": lines[j].get("confidence", 0.0) or 0.0,
                    }
                )
                j += 1
                break  # stop after first value to prevent cross-column bleed
            else:
                # Skip noise between label and value:
                #   - blank / empty lines
                #   - line reference codes  e.g. "1a", "2b", "5"
                #   - dotted leaders / pure punctuation  e.g. ". . . . ."
                # All three appear frequently in IRS form layouts (1040, 1065, etc.)
                if not next_text or _is_line_reference_cell(next_text) or _is_unusable_label(next_text):
                    j += 1
                    continue
                break
 
        if values:
            rows.append(
                {
                    "label": text,
                    "values": values,
                    "page": page,
                    "label_confidence": lines[i].get("confidence", 0.0) or 0.0,
                }
            )
            i = j
        else:
            i += 1
 
    return rows
 
 
def _expand_rows_with_period_context(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Split a row with multiple values into individual rows when value period differs.
 
    Example:
      "Total assets" with:
        1500 -> Beginning of tax year
        1500 -> End of tax year
 
    becomes:
      "Total assets [Beginning of tax year]"
      "Total assets [End of tax year]"
 
    This makes the UI and mapped output clearly show 3 distinct rows when
    combined with page-1 "Total assets".
    """
    expanded: List[Dict[str, Any]] = []
 
    for row in rows:
        label = row.get("label", "")
        page = row.get("page")
        row_index = row.get("row_index")
        values = row.get("values", []) or []
 
        if not values:
            continue
 
        # If there is only one value, keep row as is
        if len(values) == 1:
            expanded.append(
                {
                    "label": label,
                    "values": values,
                    "page": page,
                    "row_index": row_index,
                }
            )
            continue
 
        # If multiple values exist, split by period.
        # Values with a period each become their own labelled row.
        # Values without a period are grouped under the bare label.
        period_values: List[Dict[str, Any]] = []
        no_period_values: List[Dict[str, Any]] = []
        for val in values:
            if val.get("period"):
                period_values.append(val)
            else:
                no_period_values.append(val)

        if period_values:
            for val in period_values:
                expanded.append(
                    {
                        "label": f"{label}{_build_value_period_suffix(val['period'])}",
                        "values": [val],
                        "page": page,
                        "row_index": row_index,
                    }
                )
            if no_period_values:
                expanded.append(
                    {
                        "label": label,
                        "values": no_period_values,
                        "page": page,
                        "row_index": row_index,
                    }
                )
        else:
            expanded.append(
                {
                    "label": label,
                    "values": values,
                    "page": page,
                    "row_index": row_index,
                }
            )
 
    return expanded
 
 
#def _dedupe_rows(
#     rows: List[Dict[str, Any]],
# ) -> List[Dict[str, Any]]:
#     """
#     Remove exact duplicates across table path and line fallback path.
 
#     Dedup key includes:
#     - normalized label
#     - page
#     - numeric value
#     - year
#     - period
#     """
#     seen = set()
#     deduped: List[Dict[str, Any]] = []
 
#     for row in rows:
#         label_norm = _normalize_label(row.get("label", ""))
#         page = row.get("page")
#         values = row.get("values", []) or []
 
#         norm_values = []
#         for v in values:
#             raw_value = v.get("value")
#             try:
#                 value_norm = round(float(raw_value), 6)
#             except Exception:
#                 value_norm = raw_value
 
#             norm_values.append(
#                 (
#                     value_norm,
#                     v.get("year"),
#                     v.get("period"),
#                 )
#             )
 
#         key = (
#             label_norm,
#             page,
#             tuple(norm_values),
#         )
 
#         if key in seen:
#             continue
#         seen.add(key)
#         deduped.append(row)
 
#     return deduped
 
def _dedupe_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove exact duplicates across table path and line fallback path.
 
    Dedup key includes:
    - normalized label
    - page
    - numeric value
    - year
    - period
 
    Improvements:
    - Order-independent value comparison
    - Skip empty or invalid rows
    """
 
    seen = set()
    deduped: List[Dict[str, Any]] = []
 
    for row in rows:
        label_norm = _normalize_label(row.get("label", ""))
        page = row.get("page")
        values = row.get("values", []) or []
 
        #  Skip rows with no values
        if not values:
            continue
 
        norm_values = []
        for v in values:
            raw_value = v.get("value")
 
            try:
                value_norm = round(float(raw_value), 6)
            except Exception:
                value_norm = raw_value
 
            norm_values.append(
                (
                    value_norm,
                    v.get("year"),
                    v.get("period"),
                )
            )
 
        #  FIX: make dedup order-independent
        key = (
            label_norm,
            page,
            tuple(sorted(norm_values)),
        )
 
        if key in seen:
            continue
 
        seen.add(key)
        deduped.append(row)
 
    return deduped
 
 
# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
 
def extract_rows_from_cu(cu_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract financial rows from a raw Azure Content Understanding result.
 
    Strategy:
      1. Use structured table cells
      2. Also use line fallback for standalone form fields
      3. Merge and dedupe
 
    Returns list of:
        {
          "label": str,
          "values": [{"value": float, "year": int|None, "period": str|None}],
          "page": int
        }
    """
    rows: List[Dict[str, Any]] = []
 
    # Path 1: structured table cells
    cells = ContentUnderstandingClient.extract_table_cells_with_confidence(
        cu_json,
        aggregate_mode="mean",
    )
 
    table_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        key = (_safe_int(cell.get("page"), 0), _safe_int(cell.get("table_index"), 0))
        table_groups[key].append(cell)
 
    table_rows: List[Dict[str, Any]] = []
    for key in sorted(table_groups.keys()):
        table_rows.extend(_rows_from_cell_group(table_groups[key]))
 
    table_rows = _expand_rows_with_period_context(table_rows)
 
    # Path 2: line fallback
    lines = ContentUnderstandingClient.extract_lines_with_confidence(
        cu_json,
        aggregate_mode="mean",
    )
    line_rows = _rows_from_lines_fallback(lines)
 
    # Merge instead of returning early
    rows.extend(table_rows)
    rows.extend(line_rows)
 
    #  NEW: Filter invalid labels
    # Remove exact duplicates but keep distinct page/period values
    filtered_rows: List[Dict[str, Any]] = []
 
    for row in rows:
        label = row.get("label", "")
        if _is_valid_financial_label(label):
            filtered_rows.append(row)
 
    # Deduplicate after filtering
    rows = _dedupe_rows(filtered_rows)

    # Remove exact duplicates but keep distinct page/period values
    #rows = _dedupe_rows(rows)
    return rows


def extract_all_candidates(cu_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract ALL label+value pairs from a raw Azure Content Understanding result.

    Unlike extract_rows_from_cu(), this function:
    - Does NOT group values under rows — returns one dict per label+value pair
    - Does NOT filter by value magnitude or formatting
    - Does NOT perform any COA matching
    - Is purely lossless — every valid label+numeric-value pair survives

    This flat candidate list is the input for the COA-driven pipeline in
    statement_mapper.map_coa_to_document(), where each COA entry searches
    this list for its best match.

    Returns a flat list of:
        {
          "label":            str,        raw label text from the document
          "value":            float,      parsed numeric value
          "original_value":   str,        raw text as it appeared in the document
          "page":             int,        page number (1-based)
          "year":             int|None,   fiscal year detected from column header
          "period":           str|None,   e.g. "Beginning of tax year"
          "confidence":       float,      OCR confidence of the value cell (0.0–1.0)
          "label_confidence": float,      OCR confidence of the label cell (0.0–1.0)
        }
    """
    candidates: List[Dict[str, Any]] = []

    # ── Path 1: structured table cells (primary source) ──────────────────
    cells = ContentUnderstandingClient.extract_table_cells_with_confidence(
        cu_json,
        aggregate_mode="mean",
    )

    table_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        key = (_safe_int(cell.get("page"), 0), _safe_int(cell.get("table_index"), 0))
        table_groups[key].append(cell)

    for key in sorted(table_groups.keys()):
        for row in _rows_from_cell_group(table_groups[key]):
            label = row.get("label", "")
            if not _is_valid_financial_label(label):
                continue
            for v in row.get("values", []):
                candidates.append({
                    "label":          label,
                    "value":          v["value"],
                    "original_value": v.get("original_value", ""),
                    "page":           row.get("page"),
                    "year":           v.get("year"),
                    "period":         v.get("period"),
                    "confidence":     v.get("confidence", 0.0),
                    "label_confidence": row.get("label_confidence", 0.0),
                })

    # ── Path 2: line fallback (catches fields outside tables) ────────────
    lines = ContentUnderstandingClient.extract_lines_with_confidence(
        cu_json,
        aggregate_mode="mean",
    )
    for row in _rows_from_lines_fallback(lines):
        label = row.get("label", "")
        if not _is_valid_financial_label(label):
            continue
        for v in row.get("values", []):
            candidates.append({
                "label":          label,
                "value":          v["value"],
                "original_value": v.get("original_value", ""),
                "page":           row.get("page"),
                "year":           v.get("year"),
                "period":         v.get("period"),
                "confidence":     v.get("confidence", 0.0),
                "label_confidence": row.get("label_confidence", 0.0),
            })

    return candidates