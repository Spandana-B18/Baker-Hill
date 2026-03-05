"""
Streamlit app: Document Analyser
Uses Azure Content Understanding prebuilt-layout to extract content + confidence scores.
"""
 
import json
import os
 
import streamlit as st
from dotenv import load_dotenv
 
from conf_score import ContentUnderstandingClient, ContentUnderstandingError
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
load_dotenv()
 
ENDPOINT = os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT", "")
API_KEY = os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY", "")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
ANALYZER_ID = "prebuilt-layout"
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
 
# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
 
st.set_page_config(
    page_title="Document Intelligence — Content Understanding",
    page_icon="📄",
    layout="wide",
)
 
st.title("📄 Document Analyser")
st.caption(
    "Powered by **Azure Content Understanding** (`prebuilt-layout`) — "
    "upload a document to extract structured content with confidence scores."
)
 
# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
 
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "heif"],
    help=f"Maximum file size: {MAX_FILE_MB} MB",
)
 
# ---------------------------------------------------------------------------
# Analysis trigger
# ---------------------------------------------------------------------------
 
if uploaded_file:
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    file_bytes = uploaded_file.read()
    file_mb = len(file_bytes) / (1024 * 1024)
 
    col1, col2 = st.columns([3, 1])
    col1.info(f"**{uploaded_file.name}**  —  {file_mb:.2f} MB  —  {file_ext.upper()}")
 
    if file_mb > MAX_FILE_MB:
        st.error(f"File exceeds the {MAX_FILE_MB} MB limit. Please upload a smaller file.")
        st.stop()
 
    if col2.button("🔍 Analyse Document", type="primary", use_container_width=True):
        content_type = CONTENT_TYPE_MAP.get(file_ext, "application/octet-stream")
 
        with st.spinner("Submitting document to Azure Content Understanding…"):
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
                st.success("Analysis complete!")
            except ContentUnderstandingError as exc:
                st.error(f"Analysis failed: {exc}")
                st.stop()
 
# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
 
if "raw_result" in st.session_state:
    raw_result: dict = st.session_state["raw_result"]
 
    st.divider()
    json_str = json.dumps(raw_result, indent=2, default=str)
 
    st.subheader("📋 Full API Response JSON")
    st.code(json_str, language="json", line_numbers=True)
 
    st.download_button(
        "⬇️ Download JSON",
        data=json_str.encode(),
        file_name="content_understanding_result.json",
        mime="application/json",
    )
 
 