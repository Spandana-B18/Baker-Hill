from config import AZURE_CONTENT_UNDERSTANDING_ENDPOINT, AZURE_CONTENT_UNDERSTANDING_KEY, CONTENT_UNDERSTANDING_ANALYZER_ID
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentunderstanding import ContentUnderstandingClient    

def content_understanding_ir(pdf_bytes: bytes) -> dict:
    if not AZURE_CONTENT_UNDERSTANDING_ENDPOINT or not AZURE_CONTENT_UNDERSTANDING_KEY:
        raise RuntimeError("Missing AZURE_CONTENT_UNDERSTANDING_ENDPOINT or AZURE_CONTENT_UNDERSTANDING_KEY")

    client = ContentUnderstandingClient(
        endpoint=AZURE_CONTENT_UNDERSTANDING_ENDPOINT,
        credential=AzureKeyCredential(AZURE_CONTENT_UNDERSTANDING_KEY),
    )

    poller = client.begin_analyze_binary(
        analyzer_id=CONTENT_UNDERSTANDING_ANALYZER_ID,
        binary_input=pdf_bytes,
    )
    result = poller.result()

    if not result.contents:
        raise RuntimeError("Content Understanding returned no contents")

    content = result.contents[0]
    markdown = getattr(content, "markdown", "") or ""

    # Collect OCR word-level confidence from Content Understanding (when words are returned)
    all_confidences: list[float] = []
    confidence_per_page: list[dict] = []

    pages = []
    for p in getattr(content, "pages", []) or []:
        page_num = getattr(p, "page_number", None)
        pages.append(
            {
                "page_number": page_num,
                "width": getattr(p, "width", None),
                "height": getattr(p, "height", None),
            }
        )
        words = getattr(p, "words", []) or []
        page_confs = [float(getattr(w, "confidence", 0) or 0) for w in words if getattr(w, "confidence", None) is not None]
        if page_confs:
            all_confidences.extend(page_confs)
            confidence_per_page.append({
                "page_number": page_num,
                "word_count": len(page_confs),
                "avg_confidence": round(sum(page_confs) / len(page_confs), 4),
                "min_confidence": round(min(page_confs), 4),
                "max_confidence": round(max(page_confs), 4),
            })

    tables = []
    for t in getattr(content, "tables", []) or []:
        cells = []
        for c in getattr(t, "cells", []) or []:
            brs = []
            for br in getattr(c, "bounding_regions", []) or []:
                brs.append({"page": getattr(br, "page_number", None), "polygon": getattr(br, "polygon", None)})
            cells.append(
                {
                    "row": getattr(c, "row_index", None),
                    "col": getattr(c, "column_index", None),
                    "row_span": getattr(c, "row_span", 1),
                    "col_span": getattr(c, "column_span", 1),
                    "text": (getattr(c, "content", "") or "").strip(),
                    "kind": getattr(c, "kind", None),
                    "bounding_regions": brs,
                }
            )

        t_brs = []
        for br in getattr(t, "bounding_regions", []) or []:
            t_brs.append({"page": getattr(br, "page_number", None), "polygon": getattr(br, "polygon", None)})

        tables.append(
            {
                "row_count": getattr(t, "row_count", None),
                "col_count": getattr(t, "column_count", None),
                "bounding_regions": t_brs,
                "cells": cells,
            }
        )

    # Summary of OCR confidence (0–1) from Content Understanding word-level scores
    content_understanding_confidence: dict = {}
    if all_confidences:
        content_understanding_confidence = {
            "source": "content_understanding_ocr",
            "word_count": len(all_confidences),
            "avg_confidence": round(sum(all_confidences) / len(all_confidences), 4),
            "min_confidence": round(min(all_confidences), 4),
            "max_confidence": round(max(all_confidences), 4),
            "per_page": confidence_per_page,
        }

    return {
        "analyzer_id": CONTENT_UNDERSTANDING_ANALYZER_ID,
        "content_format": "markdown",
        "markdown": markdown,
        "pages": pages,
        "tables": tables,
        "content_understanding_confidence": content_understanding_confidence if content_understanding_confidence else None,
    }