import streamlit as st
import json
import uuid
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, BLOB_INPUT_CONTAINER, BLOB_OUTPUT_CONTAINER, BLOB_LOG_CONTAINER, CONTENT_UNDERSTANDING_ANALYZER_ID
from storage.blob_storage import ensure_container, blob_upload_bytes, blob_upload_json
from services.content_understanding import content_understanding_ir
from app.llm_pipeline import llm_dynamic_json, llm_business_validation
from app.validation import deterministic_validate   
from config import now_stamp

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Baker Hill POC", layout="wide")
st.title("Baker Hill POC: Blob → Content Understanding → Azure OpenAI → Validation")

with st.sidebar:
    st.header("Settings")
    st.write("Analyzer:", CONTENT_UNDERSTANDING_ANALYZER_ID)
    run_business = st.checkbox("Run LLM business validation", value=True)
    user_hint = st.text_input("Hint (optional)", value="tax or financial, and form name if known")
    chunk_chars_extract = st.slider("LLM chunk size for extraction (chars)", 8000, 24000, 18000, 1000)
    chunk_chars_validate = st.slider("LLM chunk size for validation (chars)", 6000, 16000, 12000, 1000)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
run_btn = st.button("Run pipeline", type="primary", disabled=(uploaded is None))

if run_btn:
    if not AZURE_STORAGE_CONNECTION_STRING:
        st.error("Missing AZURE_STORAGE_CONNECTION_STRING")
        st.stop()

    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    ensure_container(blob_service, BLOB_INPUT_CONTAINER)
    ensure_container(blob_service, BLOB_OUTPUT_CONTAINER)
    ensure_container(blob_service, BLOB_LOG_CONTAINER)

    doc_id = uuid.uuid4().hex
    stamp = now_stamp()
    base = f"{stamp}_{doc_id}"

    pdf_blob = f"{base}.pdf"
    ir_blob = f"{base}.content_understanding_ir.json"
    extracted_blob = f"{base}.extracted.json"
    final_blob = f"{base}.final.json"

    pdf_bytes = uploaded.getvalue()

    prog = st.progress(0)
    status = st.empty()

    status.write("Step 1: Upload PDF to Blob")
    blob_upload_bytes(blob_service, BLOB_INPUT_CONTAINER, pdf_blob, pdf_bytes, "application/pdf")
    prog.progress(15)

    status.write("Step 2: Content Understanding (OCR + layout + tables)")
    try:
        ir = content_understanding_ir(pdf_bytes)
    except Exception as ex:
        st.error(f"Content Understanding failed: {ex}")
        st.stop()

    # Do not save markdown/IR to blob for now
    # blob_upload_json(blob_service, BLOB_LOG_CONTAINER, ir_blob, ir)
    prog.progress(45)

    status.write("Step 3: Azure OpenAI dynamic JSON (chunked)")
    try:
        extracted = llm_dynamic_json(ir, user_hint=user_hint, chunk_chars=chunk_chars_extract)
    except Exception as ex:
        st.error(f"Dynamic JSON failed: {ex}")
        st.stop()

    blob_upload_json(blob_service, BLOB_LOG_CONTAINER, extracted_blob, extracted)
    prog.progress(70)

    business_report = {}
    if run_business:
        status.write("Step 4: LLM business validation (chunked)")
        try:
            business_report = llm_business_validation(ir, extracted, chunk_chars=chunk_chars_validate)
        except Exception as ex:
            business_report = {
                "issues": [
                    {
                        "path": "",
                        "severity": "medium",
                        "description": str(ex),
                        "suggested_action": "Check logs",
                        "evidence": "",
                    }
                ],
                "overall_risk": "medium",
            }

    prog.progress(85)

    status.write("Step 5: Deterministic validation")
    det_report = deterministic_validate(extracted)

    extracted.setdefault("validations", {})
    extracted["validations"]["business_validation"] = business_report
    extracted["validations"]["deterministic_validation"] = det_report

    extracted["metadata"] = extracted.get("metadata") or {}
    extracted["metadata"]["doc_id"] = doc_id
    extracted["metadata"]["created_utc"] = stamp
    extracted["metadata"]["source_blob"] = pdf_blob
    extracted["metadata"]["ir_blob"] = ir_blob

    # Add Content Understanding (OCR) confidence to final JSON when available
    cu_conf = ir.get("content_understanding_confidence")
    if cu_conf is not None:
        extracted.setdefault("confidence", {})["content_understanding"] = cu_conf

    blob_upload_json(blob_service, BLOB_OUTPUT_CONTAINER, final_blob, extracted)
    prog.progress(100)
    status.write("Done")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Content Understanding markdown preview")
        markdown_text = (ir.get("markdown") or "").strip()
        st.code(markdown_text[:6000])
        st.write("Pages detected:", len(ir.get("pages") or []))
        st.write("Tables detected:", len(ir.get("tables") or []))
        st.download_button(
            label="Download markdown",
            data=markdown_text.encode("utf-8"),
            file_name=f"{base}.content_understanding.md",
            mime="text/markdown",
            key="download_cu_md",
        )

    with c2:
        st.subheader("Final JSON (preview)")
        json_str = json.dumps(extracted, ensure_ascii=False, indent=2)
        preview_chars = 4000
        st.code(json_str[:preview_chars] + ("…" if len(json_str) > preview_chars else ""))
        st.caption(f"Showing first {min(preview_chars, len(json_str)):,} of {len(json_str):,} characters")

    st.info(f"Blob outputs: input={pdf_blob} logs={ir_blob}, {extracted_blob} output={final_blob}")
    st.subheader("Downloads")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download markdown (.md)",
            data=markdown_text.encode("utf-8"),
            file_name=f"{base}.content_understanding.md",
            mime="text/markdown",
            key="download_markdown_btn",
        )
    with dl_col2:
        st.download_button(
            label="Download final JSON",
            data=json.dumps(extracted, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=final_blob,
            mime="application/json",
            key="download_json_btn",
        )