# streamlit_app.py
import json
import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from jsonschema import Draft202012Validator

from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

from azure.ai.contentunderstanding import ContentUnderstandingClient
from openai import AzureOpenAI

load_dotenv()


def env_get(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v


# ============================================================
# Azure OpenAI
# ============================================================
AZURE_OPENAI_ENDPOINT = env_get("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_KEY = env_get("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = env_get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env_get("AZURE_OPENAI_API_VERSION", "2025-04-14")

# ============================================================
# Content Understanding
# ============================================================
AZURE_CONTENT_UNDERSTANDING_ENDPOINT = env_get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT").rstrip("/")
AZURE_CONTENT_UNDERSTANDING_KEY = env_get("AZURE_CONTENT_UNDERSTANDING_KEY")
CONTENT_UNDERSTANDING_ANALYZER_ID = env_get("CONTENT_UNDERSTANDING_ANALYZER_ID", "prebuilt-layout")

# ============================================================
# Azure Blob
# ============================================================
AZURE_STORAGE_CONNECTION_STRING = env_get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_INPUT_CONTAINER = env_get("BLOB_INPUT_CONTAINER", "input-documents")
BLOB_OUTPUT_CONTAINER = env_get("BLOB_OUTPUT_CONTAINER", "output-json")
BLOB_LOG_CONTAINER = env_get("BLOB_LOG_CONTAINER", "logfiles")


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


from blob_utils import (
    ensure_container,
    blob_upload_bytes,
    blob_upload_json,
    list_blobs,
    blob_download_bytes,
)


def azure_openai_client() -> AzureOpenAI:
    missing = []
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_KEY:
        missing.append("AZURE_OPENAI_KEY")
    if not AZURE_OPENAI_DEPLOYMENT:
        missing.append("AZURE_OPENAI_DEPLOYMENT")
    if missing:
        raise RuntimeError(f"Missing Azure OpenAI env vars: {', '.join(missing)}")

    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


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


def build_table_snippet(ir: dict, max_tables: int = 6, max_cells: int = 180) -> str:
    parts = []
    tables = ir.get("tables", []) or []
    for ti, t in enumerate(tables[:max_tables], start=1):
        parts.append(f"TABLE {ti} rows={t.get('row_count')} cols={t.get('col_count')}")
        shown = 0
        for c in (t.get("cells") or []):
            if shown >= max_cells:
                parts.append("TABLE TRUNCATED")
                break
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            parts.append(f"r{c.get('row')} c{c.get('col')} {txt[:140]}")
            shown += 1
        parts.append("")
    return "\n".join(parts).strip()


def split_into_chunks(markdown: str, max_chars: int) -> list[str]:
    if not markdown:
        return []

    # Try page break marker split first (common in CU markdown)
    marker = "<!-- PageBreak -->"
    if marker in markdown:
        parts = markdown.split(marker)
        chunks = []
        current = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            add = p + "\n" + marker + "\n"
            if len(current) + len(add) > max_chars and current.strip():
                chunks.append(current)
                current = ""
            current += add
        if current.strip():
            chunks.append(current)
        return chunks

    # Fallback: fixed-size chunks
    return [markdown[i : i + max_chars] for i in range(0, len(markdown), max_chars) if markdown[i : i + max_chars].strip()]


def merge_envelopes(envelopes: list[dict]) -> dict:
    base = envelopes[0]

    for env in envelopes[1:]:
        # Prefer schema with higher selection confidence if present
        try:
            b_conf = float((base.get("confidence") or {}).get("schema_id_selection", 0))
            e_conf = float((env.get("confidence") or {}).get("schema_id_selection", 0))
            if e_conf > b_conf and env.get("schema"):
                base["schema"] = env["schema"]
                (base.setdefault("confidence", {}))["schema_id_selection"] = e_conf
        except Exception:
            pass

        # Payload merge
        bp = base.get("payload") or {}
        ep = env.get("payload") or {}
        if isinstance(bp, dict) and isinstance(ep, dict):
            for k, v in ep.items():
                if k not in bp or bp[k] in [None, "", [], {}]:
                    bp[k] = v
                else:
                    if isinstance(bp[k], list) and isinstance(v, list):
                        bp[k].extend(v)
        base["payload"] = bp

        # Evidence merge
        be = base.get("evidence") or {}
        ee = env.get("evidence") or {}
        if isinstance(be, dict) and isinstance(ee, dict):
            for k, v in ee.items():
                if k not in be:
                    be[k] = v
                else:
                    if isinstance(be[k], list) and isinstance(v, list):
                        be[k].extend(v)
        base["evidence"] = be

        # Confidence merge (keep max per key if numeric)
        bc = base.get("confidence") or {}
        ec = env.get("confidence") or {}
        if isinstance(bc, dict) and isinstance(ec, dict):
            for k, v in ec.items():
                try:
                    v_num = float(v)
                    b_num = float(bc.get(k, 0))
                    bc[k] = max(b_num, v_num)
                except Exception:
                    if k not in bc:
                        bc[k] = v
        base["confidence"] = bc

    base.setdefault("metadata", {})
    base["metadata"]["chunking"] = {"chunks": len(envelopes)}
    return base


# Minimal starter schemas (expand as needed)
SCHEMA_REGISTRY = {
    "tax_1040": {"type": "object", "additionalProperties": True},
    "tax_1120s": {"type": "object", "additionalProperties": True},
    "tax_1065": {"type": "object", "additionalProperties": True},
    "financial_statement": {"type": "object", "additionalProperties": True},
    "annual_report": {"type": "object", "additionalProperties": True},
}

ENVELOPE_SCHEMA = {
    "type": "object",
    "required": ["metadata", "schema", "payload", "evidence", "confidence", "validations"],
    "properties": {
        "metadata": {"type": "object"},
        "schema": {
            "type": "object",
            "required": ["schema_id", "schema_version"],
            "properties": {"schema_id": {"type": "string"}, "schema_version": {"type": "string"}},
        },
        "payload": {"type": "object"},
        "evidence": {"type": "object"},
        "confidence": {"type": "object"},
        "validations": {"type": "object"},
    },
    "additionalProperties": True,
}


def llm_dynamic_json(ir: dict, user_hint: str = "", chunk_chars: int = 18000) -> dict:
    client = azure_openai_client()

    markdown_full = (ir.get("markdown") or "").strip()
    chunks = split_into_chunks(markdown_full, max_chars=chunk_chars)
    if not chunks:
        raise RuntimeError("No markdown content to process")

    schema_ids = "\n".join(sorted(SCHEMA_REGISTRY.keys()))
    table_snip = build_table_snippet(ir)

    system_msg = (
        "You extract structured data.\n"
        "Return only a single JSON object.\n"
        "Choose schema_id from the allowed list.\n"
        "Follow the envelope shape exactly.\n"
        "Only use information present in the provided chunk.\n"
        "If unsure, use null and explain in validations.\n"
    )

    envelopes = []
    for idx, chunk in enumerate(chunks, start=1):
        user_msg = (
            f"Allowed schema_id values:\n{schema_ids}\n\n"
            "Envelope shape:\n"
            "{\n"
            '  "metadata": { },\n'
            '  "schema": { "schema_id": "", "schema_version": "" },\n'
            '  "payload": { },\n'
            '  "evidence": { },\n'
            '  "confidence": { },\n'
            '  "validations": { }\n'
            "}\n\n"
            f"User hint: {user_hint}\n"
            f"Chunk {idx} of {len(chunks)}\n\n"
            "Document markdown chunk:\n"
            f"{chunk}\n\n"
            "Global tables snippet:\n"
            f"{table_snip}\n"
        )

        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.2,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"},
        )
        envelopes.append(json.loads(resp.choices[0].message.content))

    return merge_envelopes(envelopes)


def llm_business_validation(ir: dict, extracted: dict, chunk_chars: int = 12000) -> dict:
    client = azure_openai_client()

    markdown_full = (ir.get("markdown") or "").strip()
    chunks = split_into_chunks(markdown_full, max_chars=chunk_chars)
    if not chunks:
        return {"issues": [], "overall_risk": "low"}

    system_msg = (
        "You validate extracted business data.\n"
        "Return JSON only.\n"
        "Find inconsistencies, missing fields, suspicious values.\n"
        "Tie each issue to evidence from the document chunk.\n"
    )

    all_issues = []
    overall = "low"

    for idx, chunk in enumerate(chunks, start=1):
        user_msg = (
            f"Chunk {idx} of {len(chunks)}\n\n"
            "Document markdown chunk:\n"
            f"{chunk}\n\n"
            "Extracted JSON:\n"
            f"{json.dumps(extracted, ensure_ascii=False)}\n\n"
            "Return JSON with keys:\n"
            "issues: list of {path, severity, description, suggested_action, evidence}\n"
            "overall_risk: low or medium or high\n"
        )

        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.1,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)

        all_issues.extend(out.get("issues", []) or [])
        r = (out.get("overall_risk") or "low").lower()
        if r == "high":
            overall = "high"
        elif r == "medium" and overall != "high":
            overall = "medium"

    return {"issues": all_issues, "overall_risk": overall}


def deterministic_validate(extracted: dict) -> dict:
    errors = []

    v_env = Draft202012Validator(ENVELOPE_SCHEMA)
    for e in v_env.iter_errors(extracted):
        errors.append({"type": "envelope_schema", "path": list(e.absolute_path), "message": e.message})

    schema_id = ((extracted.get("schema") or {}).get("schema_id") or "").strip()
    payload = extracted.get("payload") or {}

    if not schema_id:
        errors.append({"type": "schema_id", "path": ["schema", "schema_id"], "message": "schema_id is missing"})
    elif schema_id not in SCHEMA_REGISTRY:
        errors.append({"type": "schema_id", "path": ["schema", "schema_id"], "message": "schema_id not in registry"})
    else:
        v_payload = Draft202012Validator(SCHEMA_REGISTRY[schema_id])
        for e in v_payload.iter_errors(payload):
            errors.append({"type": "payload_schema", "path": ["payload"] + list(e.absolute_path), "message": e.message})

    return {"status": "pass" if not errors else "fail", "errors": errors}


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

    # Output JSON uses same base name as input file, with .json extension
    input_name = (uploaded.name or "document").strip() or "document"
    output_base = os.path.splitext(input_name)[0]
    final_blob = f"{output_base}.json"

    pdf_blob = f"{base}.pdf"
    ir_blob = f"{base}.content_understanding_ir.json"
    extracted_blob = f"{base}.extracted.json"

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

    blob_upload_json(blob_service, BLOB_LOG_CONTAINER, ir_blob, ir)
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

    blob_upload_json(blob_service, BLOB_OUTPUT_CONTAINER, final_blob, extracted)
    prog.progress(100)
    status.write("Done")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Content Understanding markdown preview")
        st.code((ir.get("markdown") or "")[:6000])
        st.write("Pages detected:", len(ir.get("pages") or []))
        st.write("Tables detected:", len(ir.get("tables") or []))

    with c2:
        st.subheader("Final JSON")
        st.json(extracted)

    st.info(f"Blob outputs: input={pdf_blob} logs={ir_blob}, {extracted_blob} output={final_blob}")
    st.download_button(
        label="Download final JSON",
        data=json.dumps(extracted, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=final_blob,
        mime="application/json",
    )