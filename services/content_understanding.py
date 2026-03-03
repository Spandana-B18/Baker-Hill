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

    pages = []
    for p in getattr(content, "pages", []) or []:
        pages.append(
            {
                "page_number": getattr(p, "page_number", None),
                "width": getattr(p, "width", None),
                "height": getattr(p, "height", None),
            }
        )

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

    return {
        "analyzer_id": CONTENT_UNDERSTANDING_ANALYZER_ID,
        "content_format": "markdown",
        "markdown": markdown,
        "pages": pages,
        "tables": tables,
    }