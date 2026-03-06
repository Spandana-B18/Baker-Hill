"""
Streamlit app for:
1. Uploading a document
2. Running Azure Content Understanding
3. Displaying raw extractor JSON in a scrollable box
4. Generating final tax JSON with field and parent confidence_score
5. Displaying final JSON in a scrollable box
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
    default_analyzer_id,
    generate_tax_1120s_envelope,
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

st.set_page_config(
    page_title="Content Understanding Tax Extractor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Content Understanding Tax Extractor")
st.caption(
    "Upload a document, run Azure Content Understanding, then generate final tax JSON "
    "with field and parent confidence_score."
)

with st.sidebar:
    st.header("Viewer")
    json_box_height = st.slider(
        "JSON viewer height",
        min_value=220,
        max_value=1200,
        value=420,
        step=20,
        help="Controls the height of the scroll box used for JSON display.",
    )

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
                if "structured_json" in st.session_state:
                    del st.session_state["structured_json"]
                st.success("Analysis complete.")
            except ContentUnderstandingError as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()

    if col3.button("Clear", use_container_width=True):
        for key in ["raw_result", "structured_json", "file_name"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if "raw_result" in st.session_state:
    raw_result = st.session_state["raw_result"]
    file_name = st.session_state.get("file_name", "document")

    st.divider()
    st.subheader("Full Content Understanding JSON")

    raw_json_str = json.dumps(raw_result, indent=2, default=str)
    with st.container(height=json_box_height):
        st.code(raw_json_str, language="json", line_numbers=True)

    st.download_button(
        "Download extractor JSON",
        data=raw_json_str.encode("utf8"),
        file_name="content_understanding_result.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Final JSON with field and parent confidence_score")

    if st.button("Generate final JSON", type="primary", use_container_width=True):
        now = datetime.now(timezone.utc)
        created_utc = now.strftime("%Y%m%dT%H%M%SZ")
        doc_id = uuid4().hex
        source_blob = f"{created_utc}_{doc_id}.pdf"
        ir_blob = f"{created_utc}_{doc_id}.content_understanding_ir.json"

        with st.spinner("Generating final tax JSON..."):
            try:
                final_json = generate_tax_1120s_envelope(
                    raw_result=raw_result,
                    doc_id=doc_id,
                    created_utc=created_utc,
                    source_blob=source_blob,
                    ir_blob=ir_blob,
                    cu_analyzer_id=ANALYZER_ID,
                )
                st.session_state["structured_json"] = final_json
                st.success("Final JSON generated.")
            except LLMError as exc:
                st.error(f"LLM generation failed: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

if "structured_json" in st.session_state:
    structured = st.session_state["structured_json"]

    st.divider()
    st.subheader("Final JSON with field and parent confidence_score")

    structured_str = json.dumps(structured, indent=2, ensure_ascii=False)
    with st.container(height=json_box_height):
        st.code(structured_str, language="json", line_numbers=True)

    st.download_button(
        "Download final JSON",
        data=structured_str.encode("utf8"),
        file_name="final_tax_1120s.json",
        mime="application/json",
        use_container_width=True,
    )