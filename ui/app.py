"""
app.py

This version:
1. Uploads the original file to BLOB_INPUT_CONTAINER
2. Runs Azure Content Understanding
3. Saves raw extracted JSON to BLOB_OUTPUT_CONTAINER
4. Saves normalized JSON to BLOB_OUTPUT_CONTAINER
5. Generates embeddings for chunk documents
6. Creates the correct Azure AI Search index only if it does not already exist
7. Indexes chunk documents into the existing or newly created Azure AI Search index
8. Saves a small run log to BLOB_LOG_CONTAINER
9. Shows previews on screen
10. Provides a download option for the full raw extracted JSON
11. Uses input file name plus timestamp for downloaded JSON file names
"""

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from conf_score import (
    ContentUnderstandingClient,
    ContentUnderstandingError,
    default_analyzer_id,
)
from storage import (
    upload_bytes_to_blob,
    upload_json_to_blob,
    download_blob_bytes,
)
from transform import (
    build_raw_preview,
    build_normalized_document,
    build_index_documents,
    make_output_json_filename,
)
from indexer import AzureAISearchIndexer, SearchIndexerError


ENDPOINT = os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
API_KEY = os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY", "")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
ANALYZER_ID = os.getenv("AZURE_ANALYZER_ID", default_analyzer_id())
MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "")

BLOB_INPUT_CONTAINER = os.getenv("BLOB_INPUT_CONTAINER", "input-documents")
BLOB_OUTPUT_CONTAINER = os.getenv("BLOB_OUTPUT_CONTAINER", "output-json")
BLOB_LOG_CONTAINER = os.getenv("BLOB_LOG_CONTAINER", "logfiles")

AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", "")
AZURE_SEARCH_TAX_INDEX = os.getenv("AZURE_SEARCH_TAX_INDEX", "tax-documents-index")
AZURE_SEARCH_FINANCIAL_INDEX = os.getenv("AZURE_SEARCH_FINANCIAL_INDEX", "financial-documents-index")
AZURE_SEARCH_GENERIC_INDEX = os.getenv("AZURE_SEARCH_GENERIC_INDEX", "generic-documents-index")
AZURE_SEARCH_VECTOR_FIELD = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "content_vector")
AZURE_SEARCH_VECTOR_DIMENSIONS = os.getenv("AZURE_SEARCH_VECTOR_DIMENSIONS", "3072")

CONTENT_TYPE_MAP = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "heif": "image/heif",
}


def upload_run_log(log_data: dict, blob_name: str) -> str:
    return upload_json_to_blob(
        container_name=BLOB_LOG_CONTAINER,
        blob_name=blob_name,
        data=log_data,
    )


st.set_page_config(
    page_title="Content Understanding + Azure AI Search Indexer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Content Understanding + Azure AI Search Indexer")
st.caption(
    "Upload a document, run Azure Content Understanding, save source and JSON artifacts to Azure Blob Storage, "
    "prepare chunk documents, generate embeddings, and index them into Azure AI Search."
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
    st.header("Content Understanding")
    st.write(f"**Analyzer ID:** {ANALYZER_ID}")
    st.write(f"**API Version:** {API_VERSION}")
    st.write(f"**Max file size:** {MAX_FILE_MB} MB")

    st.divider()
    st.header("Azure AI Search")
    st.write(f"**Search endpoint:** {AZURE_SEARCH_SERVICE_ENDPOINT or 'not set'}")
    st.write(f"**Tax index:** {AZURE_SEARCH_TAX_INDEX}")
    st.write(f"**Financial index:** {AZURE_SEARCH_FINANCIAL_INDEX}")
    st.write(f"**Generic index:** {AZURE_SEARCH_GENERIC_INDEX}")
    st.write(f"**Vector field:** {AZURE_SEARCH_VECTOR_FIELD}")
    st.write(f"**Vector dimensions:** {AZURE_SEARCH_VECTOR_DIMENSIONS}")

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

        with st.spinner("Uploading file, running extraction, generating embeddings, and indexing into Azure AI Search..."):
            try:
                now = datetime.now(timezone.utc)
                created_utc = now.strftime("%Y%m%dT%H%M%SZ")
                doc_id = uuid4().hex

                source_blob = f"{created_utc}_{doc_id}_{uploaded_file.name}"
                raw_json_blob = f"{created_utc}_{doc_id}.content_understanding_raw.json"
                normalized_blob = f"{created_utc}_{doc_id}.normalized_document.json"
                log_blob = f"{created_utc}_{doc_id}.run_log.json"

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

                upload_json_to_blob(
                    container_name=BLOB_OUTPUT_CONTAINER,
                    blob_name=raw_json_blob,
                    data=raw_result,
                )

                raw_preview = build_raw_preview(raw_result)

                normalized_document = build_normalized_document(
                    raw_result,
                    doc_id=doc_id,
                    created_utc=created_utc,
                    source_blob=source_blob,
                    raw_json_blob=raw_json_blob,
                    cu_analyzer_id=ANALYZER_ID,
                    source_file_name=uploaded_file.name,
                )

                upload_json_to_blob(
                    container_name=BLOB_OUTPUT_CONTAINER,
                    blob_name=normalized_blob,
                    data=normalized_document,
                )

                index_documents = build_index_documents(normalized_document)

                indexer = AzureAISearchIndexer()
                index_result = indexer.prepare_and_index_documents(
                    normalized_document=normalized_document,
                    index_documents=index_documents,
                    add_vectors=True,
                )

                create_result = index_result.get("index_create_result", {})
                create_status = create_result.get("status", "unknown")
                uploaded_count = index_result.get("result", {}).get("uploaded", 0)
                failed_count = index_result.get("result", {}).get("failed", 0)
                index_name = index_result.get("index_name", "")

                log_data = {
                    "doc_id": doc_id,
                    "created_utc": created_utc,
                    "file_name": uploaded_file.name,
                    "input_blob": source_blob,
                    "output_blob": raw_json_blob,
                    "normalized_blob": normalized_blob,
                    "analyzer_id": ANALYZER_ID,
                    "status": raw_result.get("status"),
                    "document_type": normalized_document.get("document_type"),
                    "document_subtype": normalized_document.get("document_subtype"),
                    "schema_id": normalized_document.get("schema_id"),
                    "chunk_count": normalized_document.get("chunk_count"),
                    "index_document_count": len(index_documents),
                    "index_name": index_name,
                    "index_create_status": create_status,
                    "index_uploaded": uploaded_count,
                    "index_failed": failed_count,
                    "vectorized": index_result.get("vectorized"),
                    "preview_summary": raw_preview.get("summary", {}),
                }
                upload_run_log(log_data, log_blob)

                st.session_state["doc_id"] = doc_id
                st.session_state["created_utc"] = created_utc
                st.session_state["source_blob"] = source_blob
                st.session_state["raw_json_blob"] = raw_json_blob
                st.session_state["normalized_blob"] = normalized_blob
                st.session_state["log_blob"] = log_blob
                st.session_state["file_name"] = uploaded_file.name
                st.session_state["raw_preview"] = raw_preview
                st.session_state["normalized_document"] = normalized_document
                st.session_state["index_documents"] = index_documents
                st.session_state["index_result"] = index_result

                if failed_count == 0:
                    if create_status == "already_exists":
                        st.success(
                            f"Analysis complete. Reused existing index `{index_name}` and indexed "
                            f"{uploaded_count} chunks. Input file, raw JSON, normalized JSON, and log were saved to blob storage."
                        )
                    else:
                        st.success(
                            f"Analysis complete. Created index `{index_name}` and indexed "
                            f"{uploaded_count} chunks. Input file, raw JSON, normalized JSON, and log were saved to blob storage."
                        )
                else:
                    st.warning(
                        f"Analysis completed with partial indexing. Index `{index_name}` uploaded "
                        f"{uploaded_count} chunks and failed on {failed_count} chunks."
                    )

            except ContentUnderstandingError as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()
            except SearchIndexerError as exc:
                st.error(f"Azure AI Search indexing failed: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

    if col3.button("Clear", use_container_width=True):
        for key in [
            "doc_id",
            "created_utc",
            "source_blob",
            "raw_json_blob",
            "normalized_blob",
            "log_blob",
            "file_name",
            "raw_preview",
            "normalized_document",
            "index_documents",
            "index_result",
        ]:
            st.session_state.pop(key, None)
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
    st.caption(f"Raw JSON blob: `{st.session_state.get('raw_json_blob', '')}`")
    st.caption(f"Normalized blob: `{st.session_state.get('normalized_blob', '')}`")
    st.caption(f"Log blob: `{st.session_state.get('log_blob', '')}`")

    try:
        raw_bytes = download_blob_bytes(
            BLOB_OUTPUT_CONTAINER,
            st.session_state["raw_json_blob"],
        )

        download_file_name = make_output_json_filename(
            input_file_name=st.session_state.get("file_name", "document.pdf"),
            timestamp_utc=st.session_state.get("created_utc", "unknown"),
            suffix="raw_extracted",
        )

        st.download_button(
            "Download full raw extracted JSON",
            data=raw_bytes,
            file_name=download_file_name,
            mime="application/json",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Could not download raw JSON from blob: {exc}")

if st.session_state.get("normalized_document"):
    st.divider()
    st.subheader("Normalized document")

    normalized_document_str = json.dumps(
        st.session_state["normalized_document"],
        indent=2,
        default=str,
    )

    with st.container(height=json_box_height):
        st.code(normalized_document_str, language="json", line_numbers=True)

if st.session_state.get("index_documents"):
    st.divider()
    st.subheader("Prepared index documents")

    index_documents = st.session_state["index_documents"]
    st.caption(f"Prepared {len(index_documents)} chunk documents for Azure AI Search.")

    preview_docs = index_documents[:5]
    preview_docs_str = json.dumps(preview_docs, indent=2, default=str)

    with st.container(height=min(json_box_height, 500)):
        st.code(preview_docs_str, language="json", line_numbers=True)

if st.session_state.get("index_result"):
    st.divider()
    st.subheader("Azure AI Search indexing result")

    index_result = st.session_state["index_result"]
    result = index_result.get("result", {})
    create_result = index_result.get("index_create_result", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Index name", index_result.get("index_name", ""))
    col2.metric("Chunks prepared", index_result.get("chunk_count", 0))
    col3.metric("Uploaded", result.get("uploaded", 0))
    col4.metric("Failed", result.get("failed", 0))

    st.caption(f"Document type: `{index_result.get('document_type', '')}`")
    st.caption(f"Document ID: `{index_result.get('document_id', '')}`")
    st.caption(f"Vectors added: `{index_result.get('vectorized', False)}`")
    st.caption(f"Index create status: `{create_result.get('status', 'unknown')}`")

    index_result_str = json.dumps(index_result, indent=2, default=str)
    with st.container(height=min(json_box_height, 450)):
        st.code(index_result_str, language="json", line_numbers=True)