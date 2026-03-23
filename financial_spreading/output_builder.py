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
    created_utc: str = "",
    source_blob: str = "",
) -> Dict[str, Any]:
    """
    Build the spread output dictionary.

    Parameters
    ----------
    document_name : original file name
    document_id   : UUID hex from the pipeline
    mapped_rows   : output of spread_statement()[0]
    unmatched     : output of spread_statement()[1]
    created_utc   : ISO timestamp string (optional, for provenance)
    source_blob   : input blob path   (optional, for provenance)

    Returns
    -------
    A dict matching the agreed contract:

    {
        "document_name": "financials.pdf",
        "document_id":   "abc123...",
        "document_type": "financial_statement",
        "created_utc":   "20240312T120000Z",
        "source_blob":   "input-documents/financials.pdf",
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

    return {
        # ── Manager-defined contract fields ──────────────────────────────
        "document_name": document_name,
        "document_type": "financial_statement",
        "years_detected": years_detected,
        "rows": mapped_rows,
        "unmatched_rows": unmatched,
        # ── Provenance / internal fields ─────────────────────────────────
        "document_id": document_id,
        "created_utc": created_utc,
        "source_blob": source_blob,
        "summary": {
            "mapped_row_count": len(mapped_rows),
            "unmatched_row_count": len(unmatched),
            "unique_coa_lines": len(unique_coa),
            "years_detected": years_detected,
        },
    }
