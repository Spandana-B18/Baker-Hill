"""
streamlit_app.py

This version:
1. Uploads the original file to BLOB_INPUT_CONTAINER
2. Runs Azure Content Understanding
3. Saves raw extracted JSON to BLOB_OUTPUT_CONTAINER
4. Saves a small run log to BLOB_LOG_CONTAINER
5. Shows only a preview on screen
6. Provides a download option for the full raw extracted JSON
"""

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from conf_score import (
    ContentUnderstandingClient,
    ContentUnderstandingError,
    default_analyzer_id,
)

load_dotenv()

ENDPOINT = os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
API_KEY = os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY", "")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
ANALYZER_ID = os.getenv("AZURE_ANALYZER_ID", default_analyzer_id())
MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

BLOB_INPUT_CONTAINER = os.getenv("BLOB_INPUT_CONTAINER", "input-documents")
BLOB_OUTPUT_CONTAINER = os.getenv("BLOB_OUTPUT_CONTAINER", "output-json")
BLOB_LOG_CONTAINER = os.getenv("BLOB_LOG_CONTAINER", "logfiles")

CONTENT_TYPE_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "heif": "image/heif",
}


def get_blob_service_client() -> BlobServiceClient:
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING")
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


def get_container_client(container_name: str):
    service_client = get_blob_service_client()
    container_client = service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except Exception:
        pass

    return container_client


def upload_bytes_to_blob(container_name: str, blob_name: str, data: bytes, content_type: str | None = None) -> str:
    container_client = get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    if content_type:
        from azure.storage.blob import ContentSettings
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    else:
        blob_client.upload_blob(data, overwrite=True)

    return blob_name


def upload_json_to_blob(container_name: str, blob_name: str, data: dict) -> str:
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf8")
    return upload_bytes_to_blob(
        container_name=container_name,
        blob_name=blob_name,
        data=payload,
        content_type="application/json",
    )


def download_blob_bytes(container_name: str, blob_name: str) -> bytes:
    container_client = get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def build_raw_preview(raw_result: dict) -> dict:
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


def upload_run_log(log_data: dict, blob_name: str) -> str:
    return upload_json_to_blob(
        container_name=BLOB_LOG_CONTAINER,
        blob_name=blob_name,
        data=log_data,
    )


st.set_page_config(
    page_title="Content Understanding Raw Extractor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Content Understanding Raw Extractor")
st.caption(
    "Upload a document, run Azure Content Understanding, save the source file and raw extracted JSON to Azure Blob Storage, "
    "and preview the result safely."
)

with st.sidebar:
    st.header("Viewer")
    json_box_height = st.slider(
        "JSON viewer height",
        min_value=220,
        max_value=1200,
        value=420,
        step=20,
        help="Controls the height of the preview box.",
    )

    st.divider()
    st.header("Storage")
    st.write(f"**Storage account:** {AZURE_STORAGE_ACCOUNT_NAME or 'not set'}")
    st.write(f"**Input container:** {BLOB_INPUT_CONTAINER}")
    st.write(f"**Output container:** {BLOB_OUTPUT_CONTAINER}")
    st.write(f"**Log container:** {BLOB_LOG_CONTAINER}")

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

        with st.spinner("Uploading file and submitting document to Azure Content Understanding..."):
            try:
                now = datetime.now(timezone.utc)
                created_utc = now.strftime("%Y%m%dT%H%M%SZ")
                doc_id = uuid4().hex

                source_blob = f"{created_utc}_{doc_id}_{uploaded_file.name}"
                raw_json_blob = f"{created_utc}_{doc_id}.content_understanding_raw.json"
                log_blob = f"{created_utc}_{doc_id}.run_log.json"

                # Save original input file
                upload_bytes_to_blob(
                    container_name=BLOB_INPUT_CONTAINER,
                    blob_name=source_blob,
                    data=file_bytes,
                    content_type=content_type,
                )

                client = ContentUnderstandingClient(
                    endpoint=ENDPOINT,
                    api_key=API_KEY,
                    api_version=API_VERSION,
                )

                raw_result = client.analyze_document(
                    analyzer_id=ANALYZER_ID,
                    file_bytes=file_bytes,
                    file_name=uploaded_file.name,
                    content_type=content_type,
                )

                # Save raw extracted JSON
                upload_json_to_blob(
                    container_name=BLOB_OUTPUT_CONTAINER,
                    blob_name=raw_json_blob,
                    data=raw_result,
                )

                raw_preview = build_raw_preview(raw_result)

                # Save small run log
                log_data = {
                    "doc_id": doc_id,
                    "created_utc": created_utc,
                    "file_name": uploaded_file.name,
                    "input_blob": source_blob,
                    "output_blob": raw_json_blob,
                    "analyzer_id": ANALYZER_ID,
                    "status": raw_result.get("status"),
                    "preview_summary": raw_preview.get("summary", {}),
                }
                upload_run_log(log_data, log_blob)

                st.session_state["doc_id"] = doc_id
                st.session_state["created_utc"] = created_utc
                st.session_state["source_blob"] = source_blob
                st.session_state["raw_json_blob"] = raw_json_blob
                st.session_state["log_blob"] = log_blob
                st.session_state["file_name"] = uploaded_file.name
                st.session_state["raw_preview"] = raw_preview

                st.success("Analysis complete. Input file, raw JSON, and log saved to blob storage.")
            except ContentUnderstandingError as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Storage error: {exc}")
                st.stop()

    if col3.button("Clear", use_container_width=True):
        for key in [
            "doc_id",
            "created_utc",
            "source_blob",
            "raw_json_blob",
            "log_blob",
            "raw_preview",
            "file_name",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if "raw_json_blob" in st.session_state:
    raw_preview = st.session_state.get("raw_preview", {})
    raw_preview_str = json.dumps(raw_preview, indent=2, default=str)

    st.divider()
    st.subheader("Raw extracted JSON preview")

    with st.container(height=json_box_height):
        st.code(raw_preview_str, language="json", line_numbers=True)

    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Doc ID", st.session_state.get("doc_id", ""))
    meta_col2.metric("Analyzer", ANALYZER_ID)
    meta_col3.metric("Blob Saved", "Yes")

    st.caption(f"Input blob: `{st.session_state.get('source_blob', '')}`")
    st.caption(f"Output blob: `{st.session_state.get('raw_json_blob', '')}`")
    st.caption(f"Log blob: `{st.session_state.get('log_blob', '')}`")

    try:
        raw_bytes = download_blob_bytes(BLOB_OUTPUT_CONTAINER, st.session_state["raw_json_blob"])
        st.download_button(
            "Download full raw extracted JSON",
            data=raw_bytes,
            file_name="content_understanding_result.json",
            mime="application/json",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Could not download raw JSON from blob: {exc}")