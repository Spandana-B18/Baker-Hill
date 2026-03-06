"""
streamlit_app.py

Revised Streamlit app designed to avoid browser freezing on large financial
and tax documents.

What changed
1. It no longer renders the full raw Content Understanding JSON on screen
2. It shows a compact preview instead
3. Full raw JSON is still available as a download
4. Final structured JSON is also shown as a preview first
5. Full structured JSON is available as a download

This makes the UI much more stable for scanned plus text heavy PDFs.
"""

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from conf_score import (
    ContentUnderstandingClient,
    ContentUnderstandingError,
    LLMError,
    build_dynamic_document_envelope,
    default_analyzer_id,
)

load_dotenv()

ENDPOINT = os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
API_KEY = os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY", "")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
ANALYZER_ID = os.getenv("AZURE_ANALYZER_ID", default_analyzer_id())
MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

CONTENT_TYPE_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "heif": "image/heif",
}


def make_raw_preview(raw_result: dict) -> dict:
    pages = list(ContentUnderstandingClient.iter_pages(raw_result))
    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(raw_result, aggregate_mode="mean")
    table_cells = ContentUnderstandingClient.extract_table_cells_with_confidence(raw_result, aggregate_mode="mean")

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

    preview = {
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
    return preview


def make_structured_preview(structured: dict) -> dict:
    metadata = structured.get("metadata", {})
    schema = structured.get("schema", {})
    payload = structured.get("payload", {})

    preview = {
        "metadata": metadata,
        "schema": schema,
    }

    if isinstance(payload, dict):
        sampled_payload = {}
        count = 0
        for key, value in payload.items():
            sampled_payload[key] = value
            count += 1
            if count >= 8:
                break
        preview["payload_preview"] = sampled_payload
        preview["payload_key_count"] = len(payload)
    else:
        preview["payload_preview"] = payload

    return preview


st.set_page_config(
    page_title="Content Understanding Document Extractor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Content Understanding Document Extractor")
st.caption(
    "Upload a tax or financial document, run Azure Content Understanding, "
    "then generate a dynamic structured JSON envelope."
)

with st.sidebar:
    st.header("Viewer")
    json_box_height = st.slider(
        "JSON viewer height",
        min_value=220,
        max_value=1200,
        value=420,
        step=20,
        help="Controls the height of the scroll box used for JSON preview display.",
    )

    st.divider()
    st.header("Preview limits")
    st.caption("Large documents are shown as previews to keep the browser responsive.")

    st.divider()
    st.header("Current settings")
    st.write(f"**Analyzer ID:** {ANALYZER_ID}")
    st.write(f"**API Version:** {API_VERSION}")
    st.write(f"**Max file size:** {MAX_FILE_MB} MB")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heif"],
    help=f"Maximum file size: {MAX_FILE_MB} MB",
)

if uploaded_file:
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    file_bytes = uploaded_file.read()
    file_mb = len(file_bytes) / (1024 * 1024)

    col1, col2, col3 = st.columns([3, 1, 1])
    col1.info(f"{uploaded_file.name}  size {file_mb:.2f} MB")

    if file_mb > MAX_FILE_MB:
        st.error(f"File exceeds the {MAX_FILE_MB} MB limit.")
        st.stop()

    if col2.button("Run analysis", type="primary", use_container_width=True):
        content_type = CONTENT_TYPE_MAP.get(file_ext, "application/octet-stream")

        with st.spinner("Submitting document to Azure Content Understanding..."):
            client = ContentUnderstandingClient(
                endpoint=ENDPOINT,
                api_key=API_KEY,
                api_version=API_VERSION,
            )
            try:
                raw_result = client.analyze_document(
                    analyzer_id=ANALYZER_ID,
                    file_bytes=file_bytes,
                    file_name=uploaded_file.name,
                    content_type=content_type,
                )

                st.session_state["raw_result"] = raw_result
                st.session_state["file_name"] = uploaded_file.name
                st.session_state["raw_preview"] = make_raw_preview(raw_result)

                if "structured_json" in st.session_state:
                    del st.session_state["structured_json"]
                if "structured_preview" in st.session_state:
                    del st.session_state["structured_preview"]

                st.success("Analysis complete.")
            except ContentUnderstandingError as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

    if col3.button("Clear", use_container_width=True):
        for key in [
            "raw_result",
            "raw_preview",
            "structured_json",
            "structured_preview",
            "file_name",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if "raw_result" in st.session_state:
    raw_result = st.session_state["raw_result"]
    raw_preview = st.session_state.get("raw_preview", {})
    file_name = st.session_state.get("file_name", "document.pdf")

    st.divider()
    st.subheader("Content Understanding preview")

    preview_str = json.dumps(raw_preview, indent=2, default=str)
    with st.container(height=json_box_height):
        st.code(preview_str, language="json", line_numbers=True)

    raw_json_str = json.dumps(raw_result, indent=2, default=str)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download full extractor JSON",
            data=raw_json_str.encode("utf8"),
            file_name="content_understanding_result.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_b:
        st.caption("The on screen view is a compact preview. Download the full JSON here.")

    st.divider()
    st.subheader("Generate dynamic structured JSON envelope")

    if st.button("Generate final JSON", type="primary", use_container_width=True):
        now = datetime.now(timezone.utc)
        created_utc = now.strftime("%Y%m%dT%H%M%SZ")
        doc_id = uuid4().hex

        source_blob = f"{created_utc}_{doc_id}_{file_name}"
        ir_blob = f"{created_utc}_{doc_id}.content_understanding_ir.json"

        with st.spinner("Generating structured JSON envelope..."):
            try:
                final_json = build_dynamic_document_envelope(
                    raw_result=raw_result,
                    doc_id=doc_id,
                    created_utc=created_utc,
                    source_blob=source_blob,
                    ir_blob=ir_blob,
                    cu_analyzer_id=ANALYZER_ID,
                )
                st.session_state["structured_json"] = final_json
                st.session_state["structured_preview"] = make_structured_preview(final_json)
                st.success("Structured JSON generated.")
            except LLMError as exc:
                st.error(f"LLM generation failed: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

if "structured_json" in st.session_state:
    structured = st.session_state["structured_json"]
    structured_preview = st.session_state.get("structured_preview", {})

    st.divider()
    st.subheader("Final structured JSON preview")

    metadata = structured.get("metadata", {})
    schema = structured.get("schema", {})

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Document Type", metadata.get("document_type", "unknown"))
    col_b.metric("Subtype", metadata.get("document_subtype", "generic"))
    col_c.metric("Schema", schema.get("schema_id", "unknown"))

    structured_preview_str = json.dumps(structured_preview, indent=2, ensure_ascii=False)
    with st.container(height=json_box_height):
        st.code(structured_preview_str, language="json", line_numbers=True)

    structured_str = json.dumps(structured, indent=2, ensure_ascii=False)

    col_x, col_y = st.columns(2)
    with col_x:
        st.download_button(
            "Download final JSON",
            data=structured_str.encode("utf8"),
            file_name="structured_document_output.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_y:
        st.caption("The on screen view is a compact preview. Download the full structured JSON here.")