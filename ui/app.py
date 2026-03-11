"""
ui/app.py

This version:
1. Reads documents from Azure Blob input container
2. Runs Azure Content Understanding
3. Saves raw extracted JSON to BLOB_OUTPUT_CONTAINER
4. Saves normalized JSON to BLOB_OUTPUT_CONTAINER
5. Builds chunk documents
6. Creates the correct Azure AI Search index if needed
7. Indexes chunk documents into Azure AI Search
8. Saves a run log to BLOB_LOG_CONTAINER
9. Shows previews on screen
10. Provides download options for raw and normalized JSON
11. Uses retrieval-based Q&A in the second tab
12. Filters retrieval to the current document_id
13. Loads UI styling from a separate styles.css file
"""

import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from core.conf_score import (
    ContentUnderstandingClient,
    ContentUnderstandingError,
    default_analyzer_id,
)
from ingest.storage import (
    upload_json_to_blob,
    download_blob_bytes,
    get_container_client,
)
from ingest.transform import (
    build_raw_preview,
    build_normalized_document,
    build_index_documents,
    make_output_json_filename,
    compute_confidence_summary,
)
from core.indexer import AzureAISearchIndexer, SearchIndexerError
from core.retrieval_llm import RetrievalPipeline, RetrievalError


def load_css(file_name: str) -> None:
    css_path = os.path.join(CURRENT_DIR, file_name)
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def _list_blobs(container_name: str, name_starts_with: str = ""):
    cc = get_container_client(container_name)
    kwargs = {} if not name_starts_with else {"name_starts_with": name_starts_with}
    return [b.name for b in cc.list_blobs(**kwargs)]


def _list_folders(container_name: str, prefix: str = "") -> list[str]:
    blobs = _list_blobs(container_name, prefix)
    folders: set[str] = set()
    for name in blobs:
        rest = name[len(prefix):] if prefix else name
        if "/" in rest:
            folders.add(rest.split("/")[0])
    return sorted(folders)


def _list_blobs_in_folder(container_name: str, folder: str, doc_extensions: tuple[str, ...]) -> list[str]:
    if folder:
        prefix = folder.rstrip("/") + "/"
        all_blobs = _list_blobs(container_name, prefix)
    else:
        all_blobs = [b for b in _list_blobs(container_name) if "/" not in b]
    return [b for b in all_blobs if b.lower().endswith(doc_extensions)]


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
    "tif": "image/tiff",
}


@st.cache_resource(show_spinner=False)
def get_indexer() -> AzureAISearchIndexer:
    return AzureAISearchIndexer()


@st.cache_resource(show_spinner=False)
def get_retrieval_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline()


@st.cache_resource(show_spinner=False)
def get_content_understanding_client() -> ContentUnderstandingClient:
    return ContentUnderstandingClient(
        endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )


def upload_run_log(log_data: dict, blob_name: str) -> str:
    return upload_json_to_blob(
        container_name=BLOB_LOG_CONTAINER,
        blob_name=blob_name,
        data=log_data,
    )


st.set_page_config(
    page_title="Tax and Financial Statements Extraction Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css("styles.css")

_logo_paths = [
    os.path.join(CURRENT_DIR, "css", "logo.png"),
    os.path.join(CURRENT_DIR, "assets", "logo.png"),
]
_logo_b64 = None
for _p in _logo_paths:
    try:
        if os.path.isfile(_p):
            with open(_p, "rb") as _f:
                _logo_b64 = base64.b64encode(_f.read()).decode()
            break
    except Exception:
        continue

if _logo_b64:
    st.markdown(
        f'<div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 2rem; padding: 0.75rem 0;">'
        f'<img src="data:image/png;base64,{_logo_b64}" style="height: 76px; width: auto; max-width: 200px; object-fit: contain; flex-shrink: 0;" alt="Baker Hill" />'
        f'<div style="border-left: 2px solid #e2e8f0; height: 44px;"></div>'
        f'<h1 style="margin: 0; color: #1e293b; font-weight: 600; font-size: 2rem; letter-spacing: -0.02em;">Tax and Financial Statements Extraction Agent</h1>'
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.title("Tax and Financial Statements Extraction Agent")

tab_upload, tab_ask = st.tabs(["Upload and Index", "Ask Questions"])

with tab_upload:
    with st.expander("Select from Blob Storage", expanded=True):
        st.caption(f"Pick a folder, then a document from container: `{BLOB_INPUT_CONTAINER}`")
        doc_extensions = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".heif", ".tif")
        selected_blob = None

        try:
            folders = _list_folders(BLOB_INPUT_CONTAINER)
            folder_options = [""] + folders
            selected_folder = st.selectbox(
                "Select folder",
                options=folder_options,
                format_func=lambda x: "Choose folder" if x == "" else x,
                key="blob_folder_select",
            )

            if selected_folder is not None:
                doc_blobs = _list_blobs_in_folder(BLOB_INPUT_CONTAINER, selected_folder, doc_extensions)
                if doc_blobs:
                    selected_blob = st.selectbox(
                        "Select a document",
                        options=[""] + doc_blobs,
                        format_func=lambda x: "(Choose one)" if x == "" else x,
                        key="blob_doc_select",
                    )
                else:
                    st.info("No document files in this folder.")
        except Exception as e:
            st.warning(f"Could not list blobs: {e}")

    if selected_blob:
        try:
            file_bytes = download_blob_bytes(BLOB_INPUT_CONTAINER, selected_blob)
        except Exception as e:
            st.error(f"Could not download blob: {e}")
        else:
            file_name = selected_blob.split("_", 2)[-1] if selected_blob.count("_") >= 2 else selected_blob
            file_ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            file_mb = len(file_bytes) / (1024 * 1024)

            col1, col2, col3 = st.columns([3, 1, 1])
            col1.info(f"{selected_blob}  size {file_mb:.2f} MB")

            if file_mb > MAX_FILE_MB:
                st.error(f"File exceeds the {MAX_FILE_MB} MB limit.")
            elif col2.button("Run analysis", type="primary", use_container_width=True, key="run_blob"):
                content_type = CONTENT_TYPE_MAP.get(file_ext, "application/octet-stream")
                source_blob = selected_blob

                with st.status("Running analysis pipeline...", expanded=True) as pipeline_status:
                    try:
                        progress = st.progress(0, text="Starting pipeline…")

                        now = datetime.now(timezone.utc)
                        created_utc = now.strftime("%Y%m%dT%H%M%SZ")
                        doc_id = uuid4().hex

                        raw_json_blob = f"{created_utc}_{doc_id}.content_understanding_raw.json"
                        normalized_blob = f"{created_utc}_{doc_id}.normalized_document.json"
                        log_blob = f"{created_utc}_{doc_id}.run_log.json"

                        progress.progress(
                            10,
                            text="Step 1/5 — Source already in Blob. Step 2 — Running Content Understanding…",
                        )
                        st.write("Step 1/5 — Source file already in Blob Storage.")
                        st.write("Step 2/5 — Running Azure Content Understanding (may take 30–120 s)…")

                        t0 = time.monotonic()
                        client = get_content_understanding_client()
                        raw_result = client.analyze_document(
                            analyzer_id=ANALYZER_ID,
                            file_bytes=file_bytes,
                            file_name=file_name,
                            content_type=content_type,
                        )
                        st.write(f"Step 2/5 — Extraction complete. ({time.monotonic() - t0:.1f}s)")

                        conf_summary = compute_confidence_summary(raw_result)
                        st.session_state["conf_summary"] = conf_summary

                        quality_tmp = conf_summary.get("quality", "Unknown")
                        mean_conf_tmp = conf_summary.get("mean_confidence")
                        low_pct_tmp = conf_summary.get("low_conf_pct")

                        if quality_tmp == "Low":
                            st.warning(
                                f"⚠️ Low extraction confidence ({mean_conf_tmp:.0%} mean, "
                                f"{low_pct_tmp}% of lines below 70%)."
                            )
                        elif quality_tmp == "Medium":
                            st.info(
                                f"ℹ️ Moderate extraction confidence ({mean_conf_tmp:.0%} mean, "
                                f"{low_pct_tmp}% of lines below 70%)."
                            )

                        progress.progress(40, text="Step 3–4/5 — Saving raw JSON and normalizing…")
                        st.write("Step 3–4/5 — Saving raw JSON and normalizing document…")
                        t0 = time.monotonic()

                        def _save_raw():
                            upload_json_to_blob(
                                container_name=BLOB_OUTPUT_CONTAINER,
                                blob_name=raw_json_blob,
                                data=raw_result,
                            )
                            return build_raw_preview(raw_result)

                        def _normalize_and_save():
                            nd = build_normalized_document(
                                raw_result,
                                doc_id=doc_id,
                                created_utc=created_utc,
                                source_blob=source_blob,
                                raw_json_blob=raw_json_blob,
                                cu_analyzer_id=ANALYZER_ID,
                                source_file_name=file_name,
                            )
                            upload_json_to_blob(
                                container_name=BLOB_OUTPUT_CONTAINER,
                                blob_name=normalized_blob,
                                data=nd,
                            )
                            return nd

                        with ThreadPoolExecutor(max_workers=2) as pool:
                            fut_raw = pool.submit(_save_raw)
                            fut_norm = pool.submit(_normalize_and_save)
                            raw_preview = fut_raw.result()
                            normalized_document = fut_norm.result()

                        index_documents = build_index_documents(normalized_document)

                        st.write(
                            f"Step 3–4/5 — Raw JSON saved and normalized. "
                            f"Prepared {len(index_documents)} chunk(s). ({time.monotonic() - t0:.1f}s)"
                        )

                        progress.progress(60, text="Step 5/5 — Generating embeddings and indexing…")
                        st.write("Step 5/5 — Generating embeddings and indexing…")

                        t0 = time.monotonic()
                        indexer = get_indexer()
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

                        st.write(
                            f"Step 5/5 — Indexed {uploaded_count} chunk(s) into `{index_name}` "
                            f"(failed: {failed_count}). ({time.monotonic() - t0:.1f}s)"
                        )
                        progress.progress(100, text="Pipeline complete.")

                        log_data = {
                            "doc_id": doc_id,
                            "created_utc": created_utc,
                            "file_name": file_name,
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

                        st.session_state["raw_result"] = raw_result
                        st.session_state["doc_id"] = doc_id
                        st.session_state["created_utc"] = created_utc
                        st.session_state["source_blob"] = source_blob
                        st.session_state["raw_json_blob"] = raw_json_blob
                        st.session_state["normalized_blob"] = normalized_blob
                        st.session_state["log_blob"] = log_blob
                        st.session_state["file_name"] = file_name
                        st.session_state["raw_preview"] = raw_preview
                        st.session_state["normalized_document"] = normalized_document
                        st.session_state["index_documents"] = index_documents
                        st.session_state["index_result"] = index_result
                        st.session_state["doc_qa_history"] = []

                        pipeline_status.update(label="Pipeline complete.", state="complete", expanded=False)
                        st.rerun()

                    except ContentUnderstandingError as exc:
                        pipeline_status.update(
                            label="❌ Content Understanding extraction failed.",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"Analysis failed: {exc}")
                    except SearchIndexerError as exc:
                        pipeline_status.update(
                            label="❌ Azure AI Search indexing failed.",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"Azure AI Search indexing failed: {exc}")
                    except Exception as exc:
                        pipeline_status.update(label="❌ Pipeline error.", state="error", expanded=True)
                        st.error(f"Pipeline error: {exc}")

            if col3.button("Clear", use_container_width=True, key="clear_blob"):
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
                    "conf_summary",
                    "raw_result",
                    "doc_qa_history",
                ]:
                    st.session_state.pop(key, None)
                st.rerun()

    if "raw_json_blob" in st.session_state:
        conf_summary = st.session_state.get("conf_summary")
        if conf_summary:
            quality = conf_summary.get("quality", "Unknown")
            mean_conf = conf_summary.get("mean_confidence")
            min_conf = conf_summary.get("min_confidence")
            low_pct = conf_summary.get("low_conf_pct")
            total_lines = conf_summary.get("total_lines", 0)
            warning_text = conf_summary.get("warning", "")

            conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)
            conf_col1.metric("Extraction Quality", quality)
            conf_col2.metric("Mean Confidence", f"{mean_conf:.0%}" if mean_conf is not None else "N/A")
            conf_col3.metric("Min Confidence", f"{min_conf:.0%}" if min_conf is not None else "N/A")
            conf_col4.metric("Lines < 70% Conf.", f"{low_pct}%" if low_pct is not None else "N/A")

            if quality == "Low":
                st.error(f"⚠️ {warning_text}")
            elif quality == "Medium":
                st.warning(f"ℹ️ {warning_text}")

        raw_preview = st.session_state.get("raw_preview", {})
        raw_preview_str = json.dumps(raw_preview, indent=2, default=str)

        st.divider()
        st.subheader("Raw extracted JSON preview")

        with st.container(height=420):
            st.code(raw_preview_str, language="json", line_numbers=True)

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

        with st.container(height=420):
            st.code(normalized_document_str, language="json", line_numbers=True)

        try:
            norm_bytes = download_blob_bytes(
                BLOB_OUTPUT_CONTAINER,
                st.session_state["normalized_blob"],
            )

            download_file_name = make_output_json_filename(
                input_file_name=st.session_state.get("file_name", "document_normalized.pdf"),
                timestamp_utc=st.session_state.get("created_utc", "unknown"),
                suffix="normalized",
            )

            st.download_button(
                "Download full normalized JSON",
                data=norm_bytes,
                file_name=download_file_name,
                mime="application/json",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Could not download normalized JSON from blob: {exc}")

with tab_ask:
    if st.session_state.get("normalized_document"):
        file_name_current = st.session_state.get("file_name", "")

        st.markdown(
            '<p style="color: #64748b; font-size: 0.875rem; margin-bottom: 0.5rem;">Current document</p>',
            unsafe_allow_html=True,
        )

        if file_name_current:
            st.markdown(
                f'<p style="color: #1e293b; font-size: 1rem; font-weight: 600; margin-bottom: 1.5rem;">{file_name_current}</p>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p style="color: #475569; font-size: 0.9rem; margin-bottom: 1rem;">'
            "Enter your question below to query the indexed document using retrieval."
            "</p>",
            unsafe_allow_html=True,
        )

        with st.form(key="doc_qa_form", border=False):
            col_q, col_btn = st.columns([5, 1])

            with col_q:
                doc_question = st.text_input(
                    "Question",
                    placeholder="e.g. What is the leased square footage for North America?",
                    label_visibility="collapsed",
                    key="doc_qa_input",
                )

            with col_btn:
                doc_ask = st.form_submit_button("Ask", type="primary", use_container_width=True)

        if doc_ask:
            if not doc_question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching indexed chunks and generating grounded answer..."):
                    try:
                        retrieval = get_retrieval_pipeline()

                        normalized_document = st.session_state.get("normalized_document", {})
                        forced_document_type = normalized_document.get("document_type")
                        current_doc_id = st.session_state.get("doc_id")

                        filter_expression = None
                        if current_doc_id:
                            filter_expression = f"document_id eq '{current_doc_id}'"

                        qa_result = retrieval.answer(
                            query=doc_question,
                            forced_document_type=forced_document_type,
                            top_k=8,
                            filter_expression=filter_expression,
                            use_semantic=False,
                        )

                        conf_summary = st.session_state.get("conf_summary", {})
                        mean_conf = conf_summary.get("mean_confidence")
                        quality = conf_summary.get("quality", "Unknown")
                        grounded = bool(qa_result.get("grounded", False))

                        if mean_conf is None:
                            base_score = 0.50
                        else:
                            base_score = float(mean_conf)

                        if grounded:
                            base_score += 0.05
                        else:
                            base_score -= 0.20

                        if quality == "High":
                            base_score += 0.05
                        elif quality == "Low":
                            base_score -= 0.10
                        elif quality == "Unknown":
                            base_score -= 0.05

                        answer_confidence_score = max(0.0, min(0.99, round(base_score, 3)))

                        if answer_confidence_score >= 0.85:
                            answer_confidence_label = "High"
                        elif answer_confidence_score >= 0.65:
                            answer_confidence_label = "Medium"
                        else:
                            answer_confidence_label = "Low"

                        history = st.session_state.get("doc_qa_history", [])
                        history.insert(
                            0,
                            {
                                "question": doc_question,
                                "answer": qa_result.get("answer", ""),
                                "grounded": grounded,
                                "model": "retrieval_pipeline",
                                "ocr_warnings": [],
                                "doc_quality": quality,
                                "answer_confidence_score": answer_confidence_score,
                                "answer_confidence_label": answer_confidence_label,
                                "citations": qa_result.get("citations", []),
                                "retrieved_chunks": qa_result.get("results", []),
                                "index_name": qa_result.get("index_name", ""),
                                "vector_mode_used": qa_result.get("vector_mode_used", False),
                            },
                        )
                        st.session_state["doc_qa_history"] = history
                        st.rerun()

                    except RetrievalError as exc:
                        st.error(f"Retrieval Q&A failed: {exc}")
                    except Exception as exc:
                        st.error(f"Q&A failed: {exc}")

        history_items = st.session_state.get("doc_qa_history", [])
        if history_items:
            st.divider()
            st.subheader("Answers")

        for qa in history_items:
            with st.container(border=True):
                st.markdown(f"**Q:** {qa['question']}")

                meta1, meta2, meta3 = st.columns(3)
                meta1.metric(
                    "Answer Confidence",
                    f"{qa['answer_confidence_score']:.0%}" if qa.get("answer_confidence_score") is not None else "N/A",
                )
                meta2.metric(
                    "Confidence Level",
                    qa.get("answer_confidence_label", "Unknown"),
                )
                meta3.metric(
                    "Grounded",
                    "Yes" if qa.get("grounded") else "No",
                )

                st.success(qa.get("answer", ""))

                if qa.get("doc_quality"):
                    st.caption(f"Document quality used for confidence derivation: {qa['doc_quality']}")

                if qa.get("index_name"):
                    st.caption(
                        f"Index used: {qa['index_name']} | "
                        f"Vector search: {'Yes' if qa.get('vector_mode_used') else 'No'}"
                    )

                if qa.get("citations"):
                    st.markdown("**Citations**")
                    for c in qa["citations"]:
                        rank = c.get("rank", "")
                        page_number = c.get("page_number", "")
                        chunk_id = c.get("chunk_id", "")
                        source_file_name = c.get("source_file_name", "")
                        st.write(
                            f"Rank {rank} | File: {source_file_name} | Page {page_number} | Chunk {chunk_id}"
                        )

                if qa.get("retrieved_chunks"):
                    with st.expander("Retrieved chunks"):
                        for i, chunk in enumerate(qa["retrieved_chunks"], start=1):
                            page_number = chunk.get("page_number", "N/A")
                            chunk_id = chunk.get("chunk_id", "N/A")
                            score = chunk.get("@search.score")
                            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"

                            st.markdown(
                                f"**Chunk {i}** | Page: {page_number} | Chunk ID: {chunk_id} | Search score: {score_text}"
                            )
                            st.code(chunk.get("content", ""), language="text")
    else:
        st.info("Analyze a document first in the Upload and Index tab, then ask questions here.")