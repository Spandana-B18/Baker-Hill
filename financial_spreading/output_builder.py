"""
financial_spreading/output_builder.py

Assembles the final structured spread JSON that is saved to Blob
Storage and shown in the Streamlit UI.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_output(
    document_name: str,
    document_id: str,
    mapped_rows: List[Dict[str, Any]],
    unmatched: List[Dict[str, Any]],
    *,
    document_type: str = "financial_statement",
    document_subtype: str = "",
    schema_id: str = "",
    created_utc: str = "",
    source_blob: str = "",
) -> Dict[str, Any]:
    """
    Build the spread output dictionary.

    Parameters
    ----------
    document_name     : original file name
    document_id       : UUID hex from the pipeline
    mapped_rows       : output of map_coa_to_document()
    unmatched         : unmatched rows (empty list for COA-driven pipeline)
    document_type     : detected type, e.g. "tax_document" / "financial_document"
    document_subtype  : detected form, e.g. "1120s" / "1040" / "1065" / "1120"
    schema_id         : schema used, e.g. "tax_1120s" / "financial_statement"
    created_utc       : ISO timestamp string (optional, for provenance)
    source_blob       : input blob path   (optional, for provenance)

    Returns
    -------
    A dict matching the agreed contract:

    {
        "document_name":    "financials.pdf",
        "document_id":      "abc123...",
        "document_type":    "tax_document",
        "document_subtype": "1120s",
        "schema_id":        "tax_1120s",
        "created_utc":      "20240312T120000Z",
        "source_blob":      "input-documents/financials.pdf",
        "summary": {
            "mapped_row_count":    42,
            "unmatched_row_count": 3,
            "unique_coa_lines":    18,
        },
        "rows": [...],
        "unmatched_rows": [...]
    }
    """
    unique_coa = {r["chart_of_account_line"] for r in mapped_rows}

    # Collect all detected years in sorted order (None values filtered out)
    years_detected = sorted(
        {r["year"] for r in mapped_rows if r.get("year") is not None}
    )

    # Backfill rows that have no year with the document-level year.
    # Rows from pages where year appears in a column header already have
    # a year set; rows from pages without a year column header (e.g. page 1
    # form fields) get None. We fill those with the most common detected year
    # so the year column is never blank for a single-year document.
    if years_detected:
        from collections import Counter as _Counter
        _year_counts = _Counter(
            r["year"] for r in mapped_rows if r.get("year") is not None
        )
        fallback_year = _year_counts.most_common(1)[0][0]
        for r in mapped_rows:
            if r.get("year") is None:
                r["year"] = fallback_year

    return {
        # ── Manager-defined contract fields ──────────────────────────────
        "document_name":    document_name,
        "document_type":    document_type,
        "document_subtype": document_subtype,
        "schema_id":        schema_id,
        "years_detected":   years_detected,
        "rows":             mapped_rows,
        "unmatched_rows":   unmatched,
        # ── Provenance / internal fields ─────────────────────────────────
        "document_id":  document_id,
        "created_utc":  created_utc,
        "source_blob":  source_blob,
        "summary": {
            "mapped_row_count":    len(mapped_rows),
            "unmatched_row_count": len(unmatched),
            "unique_coa_lines":    len(unique_coa),
        },
    }
