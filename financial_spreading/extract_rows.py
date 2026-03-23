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
 
 
def _parse_numeric(text: str) -> Optional[float]:
    """
    Parse a cell string as a number.
    Handles: 1,234,567 | (1,234) | $1,234.56 | 1234 | 1.5%
    Returns None if not numeric.
    """
    t = (text or "").strip()
    if not t:
        return None
    t = t.replace(",", "").replace("$", "").replace("%", "")
    t = re.sub(r"^\((.+)\)$", r"-\1", t)
    try:
        return float(t)
    except (ValueError, TypeError):
        return None
 
 
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
 
 
def _rows_from_cell_group(
    cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert a group of cells (same page + table_index) into financial rows.
    Cells come from ContentUnderstandingClient.extract_table_cells_with_confidence
    so field names are: row_index, column_index, text, page.
    """
    if not cells:
        return []
 
    page_number = cells[0].get("page", 0)
 
    # Build row_map: {row_index: {col_index: text}}
    row_map: Dict[int, Dict[int, str]] = {}
    for cell in cells:
        r = _safe_int(cell.get("row_index"), 0)
        c = _safe_int(cell.get("column_index"), 0)
        row_map.setdefault(r, {})[c] = (cell.get("text") or "").strip()
 
    col_years = _detect_column_years(row_map)
    col_periods = _detect_column_periods(row_map)
 
    rows: List[Dict[str, Any]] = []
    for row_idx in sorted(row_map.keys()):
        col_map = row_map[row_idx]
        if not col_map:
            continue
 
        max_col = max(col_map.keys())
        ordered = [(i, col_map.get(i, "")) for i in range(max_col + 1)]
 
        # First non-numeric, non-line-reference cell is the label
        label = ""
        label_col = -1
        for col_idx, cell_text in ordered:
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
 
        values: List[Dict[str, Any]] = []
        for col_idx, cell_text in ordered:
            if col_idx == label_col:
                continue
            # Skip cells that look like line references (e.g. "1a", "21")
            if _is_line_reference_cell(cell_text):
                continue
 
            parsed = _parse_numeric(cell_text)
            if parsed is None:
                continue
 
            # Reject values that don't look like financial amounts
            if not _is_financial_amount(cell_text, parsed):
                continue
 
            period_label = col_periods.get(col_idx)
            value_entry = {
                "value": parsed,
                "original_value": cell_text.strip(),
                "year": col_years.get(col_idx),
                "period": period_label,
                "column_index": col_idx,
            }
            values.append(value_entry)
 
        if values:
            rows.append(
                {
                    "label": label,
                    "values": values,
                    "page": page_number,
                    "row_index": row_idx,
                }
            )
 
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
        while j < len(lines) and j <= i + 3:
            next_text = (lines[j].get("text") or "").strip()
            parsed = _parse_numeric(next_text)
            if parsed is not None:
                # Skip year-like values (e.g. "2023")
                if _YEAR_RE.fullmatch(next_text.strip()):
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
                    }
                )
                j += 1
            else:
                break
 
        if values:
            rows.append(
                {
                    "label": text,
                    "values": values,
                    "page": page,
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
 
        # If multiple values exist, split when a period is present
        split_any = False
        for val in values:
            period = val.get("period")
            if period:
                split_any = True
                expanded.append(
                    {
                        "label": f"{label}{_build_value_period_suffix(period)}",
                        "values": [val],
                        "page": page,
                        "row_index": row_index,
                    }
                )
 
        if not split_any:
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
 