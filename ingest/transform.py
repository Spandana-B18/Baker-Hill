import os
from typing import Any, Dict, List
from uuid import uuid4

from core.conf_score import (
    ContentUnderstandingClient,
    build_dynamic_document_envelope,
)
from financial_spreading.extract_rows import extract_all_candidates, extract_dependents_from_cu


def make_output_json_filename(
    input_file_name: str,
    timestamp_utc: str,
    suffix: str = "raw_extracted",
):
    base = os.path.splitext(input_file_name)[0]

    safe = "".join(
        c if c.isalnum() or c in ("_", "-", ".") else "_"
        for c in base
    )

    return f"{safe}_{timestamp_utc}_{suffix}.json"


def build_raw_preview(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    pages = list(ContentUnderstandingClient.iter_pages(raw_result))

    lines = ContentUnderstandingClient.extract_lines_with_confidence(
        raw_result,
        aggregate_mode="mean",
    )

    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(
        raw_result,
        aggregate_mode="mean",
    )

    table_cells = ContentUnderstandingClient.extract_table_cells_with_confidence(
        raw_result,
        aggregate_mode="mean",
    )

    page_summaries = []

    for page in pages[:20]:
        page_number = page.get("pageNumber", page.get("page"))

        page_summaries.append(
            {
                "page_number": page_number,
                "width": page.get("width"),
                "height": page.get("height"),
                "unit": page.get("unit"),
                "word_count": len(page.get("words", []) or []),
                "line_count": len(page.get("lines", []) or []),
                "paragraph_count": len(page.get("paragraphs", []) or []),
                "table_count": len(page.get("tables", []) or []),
            }
        )

    return {
        "status": raw_result.get("status"),
        "summary": {
            "page_count": len(pages),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "table_cell_count": len(table_cells),
        },
        "pages": page_summaries,
        "sample_lines": lines[:20],
        "sample_paragraphs": paragraphs[:10],
        "sample_table_cells": table_cells[:20],
    }


def compute_confidence_summary(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compute an overall confidence summary from extracted lines and table cells."""
    lines = ContentUnderstandingClient.extract_lines_with_confidence(
        raw_result,
        aggregate_mode="mean",
    )

    scores = [r["confidence"] for r in lines if r.get("confidence") is not None]

    if not scores:
        return {
            "mean_confidence": None,
            "min_confidence": None,
            "low_conf_pct": None,
            "total_lines": 0,
            "quality": "Unknown",
            "warning": "No confidence data available for this document.",
        }

    LOW_THRESHOLD = 0.70
    mean_conf = sum(scores) / len(scores)
    min_conf = min(scores)
    low_count = sum(1 for s in scores if s < LOW_THRESHOLD)
    low_pct = round(low_count / len(scores) * 100, 1)

    if mean_conf >= 0.85:
        quality = "High"
        warning = ""
    elif mean_conf >= 0.65:
        quality = "Medium"
        warning = (
            f"{low_pct}% of extracted lines have confidence below {int(LOW_THRESHOLD*100)}%. "
            "The document may contain scanned text, mixed fonts, or low image quality. "
            "Review extracted values carefully."
        )
    else:
        quality = "Low"
        warning = (
            f"{low_pct}% of extracted lines have confidence below {int(LOW_THRESHOLD*100)}% "
            f"(overall mean: {mean_conf:.0%}). "
            "This document likely contains handwriting, poor scan quality, or complex layouts. "
            "Extracted data may be inaccurate — manual review is strongly recommended."
        )

    return {
        "mean_confidence": round(mean_conf, 3),
        "min_confidence": round(min_conf, 3),
        "low_conf_pct": low_pct,
        "total_lines": len(scores),
        "quality": quality,
        "warning": warning,
    }


def extract_searchable_text(raw_result: Dict[str, Any]) -> str:
    lines = ContentUnderstandingClient.extract_lines_with_confidence(
        raw_result,
        aggregate_mode="mean",
    )

    parts: List[str] = []

    for row in lines:
        text = str(row.get("text", "")).strip()
        if text:
            parts.append(text)

    return "\n".join(parts)


def build_chunks_from_lines(
    raw_result: Dict[str, Any],
    *,
    document_id: str,
    max_lines_per_chunk: int = 12,
):
    lines = ContentUnderstandingClient.extract_lines_with_confidence(
        raw_result,
        aggregate_mode="mean",
    )

    grouped: Dict[int, List[Dict[str, Any]]] = {}

    for row in lines:
        page = int(row.get("page", 0) or 0)
        grouped.setdefault(page, []).append(row)

    chunks: List[Dict[str, Any]] = []

    for page_number, page_lines in grouped.items():
        current_batch: List[Dict[str, Any]] = []

        for line in page_lines:
            current_batch.append(line)
            if len(current_batch) >= max_lines_per_chunk:
                chunk = _build_chunk(
                    current_batch,
                    document_id,
                    page_number,
                )
                if chunk:
                    chunks.append(chunk)
                current_batch = []

        if current_batch:
            chunk = _build_chunk(
                current_batch,
                document_id,
                page_number,
            )
            if chunk:
                chunks.append(chunk)

    return chunks


def _build_chunk(
    current_batch,
    document_id,
    page_number,
):
    text = " ".join(
        str(x.get("text", "")).strip()
        for x in current_batch
        if str(x.get("text", "")).strip()
    )

    if not text.strip():
        return None

    confidences = [
        float(x.get("confidence") or 0.0)
        for x in current_batch
        if x.get("confidence") is not None
    ]

    avg_conf = (
        round(sum(confidences) / len(confidences), 3)
        if confidences
        else 0.0
    )

    return {
        "chunk_id": uuid4().hex,
        "document_id": document_id,
        "page_number": page_number,
        "content": text,
        "confidence_score": avg_conf,
        "chunk_type": "line_batch",
        "line_count": len(current_batch),
    }


def build_normalized_document(
    raw_result: Dict[str, Any],
    *,
    doc_id: str,
    created_utc: str,
    source_blob: str,
    raw_json_blob: str,
    cu_analyzer_id: str,
    source_file_name: str,
):
    envelope = build_dynamic_document_envelope(
        raw_result,
        doc_id=doc_id,
        created_utc=created_utc,
        source_blob=source_blob,
        ir_blob=raw_json_blob,
        cu_analyzer_id=cu_analyzer_id,
    )

    metadata = envelope.get("metadata", {})
    schema = envelope.get("schema", {})

    searchable_text = extract_searchable_text(raw_result)

    # Extract flat financial rows (label + value pairs) from all table cells
    # and line fallback. This preserves the full table structure in a flat,
    # indexable format so the spreading pipeline can read from the normalized
    # document instead of the raw CU JSON.
    try:
        financial_rows = extract_all_candidates(raw_result)
    except Exception:
        financial_rows = []

    # Extract structured dependent records (Form 1040 dependents table).
    # These are personal/non-financial fields excluded from financial_rows
    # but captured here as structured key-value objects.
    try:
        dependents = extract_dependents_from_cu(raw_result)
    except Exception:
        dependents = []

    chunks = build_chunks_from_lines(
        raw_result,
        document_id=doc_id,
    )

    normalized = {
        "document_id": doc_id,
        "document_type": metadata.get("document_type"),
        "document_subtype": metadata.get("document_subtype"),
        "title": metadata.get("document_title") or source_file_name,
        "source_file_name": source_file_name,
        "source_blob": source_blob,
        "raw_json_blob": raw_json_blob,
        "created_utc": created_utc,
        "cu_analyzer_id": cu_analyzer_id,
        "routing_confidence_score": metadata.get("routing_confidence_score"),
        "schema_id": schema.get("schema_id"),
        "schema_version": schema.get("schema_version"),
        "payload": envelope.get("payload", {}),
        "search_text": searchable_text,
        "financial_rows": financial_rows,
        "dependents": dependents,
        "chunks": chunks,
        "chunk_count": len(chunks),
    }

    return normalized


def build_index_documents(normalized_document: Dict[str, Any]):
    docs: List[Dict[str, Any]] = []

    for chunk in normalized_document.get("chunks", []):
        docs.append(
            {
                "id": chunk["chunk_id"],
                "document_id": normalized_document["document_id"],
                "chunk_id": chunk["chunk_id"],
                "title": normalized_document.get("title"),
                "content": chunk.get("content", ""),
                "document_type": normalized_document.get("document_type"),
                "document_subtype": normalized_document.get("document_subtype"),
                "schema_id": normalized_document.get("schema_id"),
                "schema_version": normalized_document.get("schema_version"),
                "source_file_name": normalized_document.get("source_file_name"),
                "source_blob": normalized_document.get("source_blob"),
                "raw_json_blob": normalized_document.get("raw_json_blob"),
                "page_number": chunk.get("page_number"),
                "chunk_type": chunk.get("chunk_type"),
                "line_count": chunk.get("line_count"),
                "confidence_score": chunk.get("confidence_score"),
                "processed_at": normalized_document.get("created_utc"),
            }
        )

    return docs

