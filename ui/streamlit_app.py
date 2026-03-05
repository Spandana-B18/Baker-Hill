# streamlit_app.py
import json
import os
import re
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from jsonschema import Draft202012Validator

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

from cu_confidence import content_understanding_confidence

load_dotenv()


def env_get(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v


# Azure OpenAI
AZURE_OPENAI_ENDPOINT = env_get("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_KEY = env_get("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = env_get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env_get("AZURE_OPENAI_API_VERSION", "2025-04-14")

# Content Understanding
AZURE_CONTENT_UNDERSTANDING_ENDPOINT = env_get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT").rstrip("/")
AZURE_CONTENT_UNDERSTANDING_KEY = env_get("AZURE_CONTENT_UNDERSTANDING_KEY")
CONTENT_UNDERSTANDING_ANALYZER_ID = env_get("CONTENT_UNDERSTANDING_ANALYZER_ID", "prebuilt-layout")

# Azure Blob
AZURE_STORAGE_CONNECTION_STRING = env_get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_INPUT_CONTAINER = env_get("BLOB_INPUT_CONTAINER", "input-documents")
BLOB_OUTPUT_CONTAINER = env_get("BLOB_OUTPUT_CONTAINER", "output-json")
BLOB_LOG_CONTAINER = env_get("BLOB_LOG_CONTAINER", "logfiles")


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def ensure_container(blob_service: BlobServiceClient, name: str) -> None:
    try:
        blob_service.create_container(name)
    except Exception:
        pass


def blob_upload_bytes(
    blob_service: BlobServiceClient,
    container: str,
    blob_name: str,
    data: bytes,
    content_type: str,
) -> None:
    bc = blob_service.get_blob_client(container=container, blob=blob_name)
    bc.upload_blob(data, overwrite=True, content_type=content_type)


def blob_upload_json(blob_service: BlobServiceClient, container: str, blob_name: str, obj: dict) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    blob_upload_bytes(blob_service, container, blob_name, data, "application/json")


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
                    # Needed for table cell OCR confidence logic in cu_confidence.py
                    "spans": [
                        {"offset": getattr(s, "offset", None), "length": getattr(s, "length", None)}
                        for s in (getattr(c, "spans", []) or [])
                        if getattr(s, "offset", None) is not None and getattr(s, "length", None) is not None
                    ],
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

    word_items = []
    for w in getattr(content, "words", []) or []:
        brs = []
        for br in getattr(w, "bounding_regions", []) or []:
            brs.append({"page": getattr(br, "page_number", None), "polygon": getattr(br, "polygon", None)})

        span_obj = getattr(w, "span", None)
        word_items.append(
            {
                "text": (getattr(w, "content", "") or "").strip(),
                "confidence": getattr(w, "confidence", None),
                "span": {
                    "offset": getattr(span_obj, "offset", None),
                    "length": getattr(span_obj, "length", None),
                },
                "bounding_regions": brs,
            }
        )

    return {
        "analyzer_id": CONTENT_UNDERSTANDING_ANALYZER_ID,
        "content_format": "markdown",
        "markdown": markdown,
        "pages": pages,
        "tables": tables,
        "words": word_items,
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

    return [
        markdown[i : i + max_chars]
        for i in range(0, len(markdown), max_chars)
        if markdown[i : i + max_chars].strip()
    ]


SCHEMA_REGISTRY = {
    "tax_1040": {"type": "object", "additionalProperties": True},
    "tax_1120s": {"type": "object", "additionalProperties": True},
    "tax_1065": {"type": "object", "additionalProperties": True},
    "financial_statement": {"type": "object", "additionalProperties": True},
    "annual_report": {"type": "object", "additionalProperties": True},
}

ENVELOPE_SCHEMA = {
    "type": "object",
    "required": ["metadata", "schema", "payload", "evidence", "validations"],
    "properties": {
        "metadata": {"type": "object"},
        "schema": {
            "type": "object",
            "required": ["schema_id", "schema_version"],
            "properties": {"schema_id": {"type": "string"}, "schema_version": {"type": "string"}},
        },
        "payload": {"type": "object"},
        "evidence": {"type": "object"},
        "validations": {"type": "object"},
    },
    "additionalProperties": True,
}


def normalize_llm_output(envelope: dict) -> dict:
    if not isinstance(envelope, dict):
        return envelope

    for k in ["confidence", "confidence_score", "confidence_level"]:
        if k in envelope:
            del envelope[k]
    return envelope


def merge_envelopes(envelopes: list[dict]) -> dict:
    base = envelopes[0]

    for env in envelopes[1:]:
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

        bv = base.get("validations") or {}
        ev = env.get("validations") or {}
        if isinstance(bv, dict) and isinstance(ev, dict):
            for k, v in ev.items():
                if k not in bv:
                    bv[k] = v
        base["validations"] = bv

        if not (base.get("schema") or {}).get("schema_id") and env.get("schema"):
            base["schema"] = env["schema"]

    base.setdefault("metadata", {})
    base["metadata"]["chunking"] = {"chunks": len(envelopes)}
    return base


def _evidence_has_key(evidence: object, key: str) -> bool:
    if not key:
        return False
    if isinstance(evidence, dict):
        if key in evidence and evidence.get(key) not in [None, "", [], {}]:
            return True
        return any(_evidence_has_key(v, key) for v in evidence.values())
    if isinstance(evidence, list):
        return any(_evidence_has_key(v, key) for v in evidence)
    return False


def _collect_table_text(ir: dict) -> str:
    parts = []
    for t in (ir.get("tables") or []):
        for c in (t.get("cells") or []):
            txt = (c.get("text") or "").strip()
            if txt:
                parts.append(txt)
    return " ".join(parts)


def _normalize_for_match(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _value_in_text(value_str: str, text: str) -> bool:
    if not value_str or not text:
        return False

    v = _normalize_for_match(value_str)
    t = _normalize_for_match(text)
    if not v:
        return False

    if v in t:
        return True

    v_digits = re.sub(r"[^\d]", "", v)
    if len(v_digits) >= 4:
        t_digits = re.sub(r"[^\d]", "", t)
        if v_digits and v_digits in t_digits:
            return True

    return False


def _page_clarity_map(ir: dict) -> dict:
    pages = ir.get("pages") or []
    words = ir.get("words") or []

    page_numbers = [p.get("page_number") for p in pages if p.get("page_number") is not None]
    if not page_numbers:
        return {1: 0.6}

    if words:
        by_page = {pn: [] for pn in page_numbers}
        for w in words:
            conf = w.get("confidence", None)
            if conf is None:
                continue
            try:
                conf_f = float(conf)
            except Exception:
                continue
            for br in (w.get("bounding_regions") or []):
                pn = br.get("page", None) or br.get("page_number", None)
                if pn in by_page:
                    by_page[pn].append(conf_f)

        clarity = {}
        for pn in page_numbers:
            vals = by_page.get(pn) or []
            if vals:
                clarity[pn] = round(max(0.0, min(1.0, sum(vals) / len(vals))), 3)
            else:
                clarity[pn] = 0.55
        return clarity

    return {pn: 0.55 for pn in page_numbers}


def _field_page_from_evidence(evidence: object, field_name: str):
    if not field_name:
        return None
    if not isinstance(evidence, dict):
        return None

    node = evidence.get(field_name, None)
    if isinstance(node, dict):
        brs = node.get("bounding_regions") or node.get("boundingRegions") or []
        if brs and isinstance(brs, list) and isinstance(brs[0], dict):
            pn = brs[0].get("page", None) or brs[0].get("page_number", None)
            return pn
    return None


def _leaf_confidence(field_name: str, value: object, base: float, evidence: object, ir: dict) -> float:
    if value in [None, "", [], {}]:
        return 0.0

    base = float(base)

    md_text = (ir.get("markdown") or "") if isinstance(ir, dict) else ""
    table_text = _collect_table_text(ir) if isinstance(ir, dict) else ""

    value_str = str(value).strip()
    in_md = _value_in_text(value_str, md_text)
    in_tbl = _value_in_text(value_str, table_text)
    has_ev = _evidence_has_key(evidence, field_name)

    page_map = _page_clarity_map(ir)
    pn = _field_page_from_evidence(evidence, field_name)

    if pn is None:
        page_quality = sum(page_map.values()) / max(1, len(page_map))
    else:
        page_quality = page_map.get(pn, 0.55)

    score = 0.15 * base + 0.55 * page_quality
    if in_md:
        score += 0.18
    if in_tbl:
        score += 0.12
    if has_ev:
        score += 0.08

    score = max(0.0, min(1.0, score))
    return round(score, 3)


def attach_field_confidence(payload, evidence, base_confidence, ir, field_name=""):
    if isinstance(payload, dict):
        if set(payload.keys()) == {"value", "confidence_score"}:
            return payload

        new_obj = {}
        child_scores = []

        for k, v in payload.items():
            wrapped = attach_field_confidence(
                v,
                evidence=evidence,
                base_confidence=base_confidence,
                ir=ir,
                field_name=str(k),
            )
            new_obj[k] = wrapped
            if isinstance(wrapped, dict) and "confidence_score" in wrapped:
                child_scores.append(wrapped["confidence_score"])

        new_obj["confidence_score"] = round(sum(child_scores) / len(child_scores), 3) if child_scores else 0.0
        return new_obj

    if isinstance(payload, list):
        wrapped_items = []
        child_scores = []

        for v in payload:
            wrapped = attach_field_confidence(
                v,
                evidence=evidence,
                base_confidence=base_confidence,
                ir=ir,
                field_name=field_name,
            )
            wrapped_items.append(wrapped)
            if isinstance(wrapped, dict) and "confidence_score" in wrapped:
                child_scores.append(wrapped["confidence_score"])

        return {
            "items": wrapped_items,
            "confidence_score": round(sum(child_scores) / len(child_scores), 3) if child_scores else 0.0,
        }

    score = _leaf_confidence(field_name, payload, base_confidence, evidence, ir)
    return {"value": payload, "confidence_score": score}


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
        "You must follow the envelope shape exactly.\n"
        "Do not add any top level keys other than: metadata, schema, payload, evidence, validations.\n"
        "Do not output a key named confidence, confidence_score, or confidence_level.\n"
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

        chunk_env = json.loads(resp.choices[0].message.content)
        chunk_env = normalize_llm_output(chunk_env)
        envelopes.append(chunk_env)

    merged = merge_envelopes(envelopes)
    merged = normalize_llm_output(merged)
    return merged


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


def extract_word_confidence_from_ir(ir: dict) -> list[dict]:
    words_data = []
    for w in (ir.get("words") or []):
        conf = w.get("confidence", None)
        try:
            conf_f = float(conf) if conf is not None else None
        except Exception:
            conf_f = None

        page = None
        polygon = None
        brs = w.get("bounding_regions") or []
        if brs and isinstance(brs, list) and isinstance(brs[0], dict):
            page = brs[0].get("page", None) or brs[0].get("page_number", None)
            polygon = brs[0].get("polygon", None)

        words_data.append(
            {
                "page": page,
                "word": w.get("text") or "",
                "confidence": round(conf_f, 4) if conf_f is not None else None,
                "polygon": polygon,
            }
        )
    return words_data


def compute_word_confidence_stats(words_data: list[dict]) -> dict:
    confs = [w["confidence"] for w in words_data if w.get("confidence") is not None]
    if not confs:
        return {"total_words": len(words_data), "avg_confidence": None, "low_confidence_count": 0}

    avg_conf = sum(confs) / len(confs)
    low_count = sum(1 for c in confs if c < 0.8)
    return {
        "total_words": len(words_data),
        "avg_confidence": round(avg_conf, 4),
        "low_confidence_count": low_count,
    }


# Streamlit UI
st.set_page_config(page_title="Baker Hill POC", layout="wide")
st.title("Baker Hill POC Blob to Content Understanding to Azure OpenAI to Validation")

with st.sidebar:
    st.header("Settings")
    st.write("CU endpoint:", AZURE_CONTENT_UNDERSTANDING_ENDPOINT)
    st.write("CU analyzer id:", repr(CONTENT_UNDERSTANDING_ANALYZER_ID))
    run_business = st.checkbox("Run LLM business validation", value=True)
    user_hint = st.text_input("Hint optional", value="tax or financial, and form name if known")
    chunk_chars_extract = st.slider("LLM chunk size for extraction chars", 8000, 24000, 18000, 1000)
    chunk_chars_validate = st.slider("LLM chunk size for validation chars", 6000, 16000, 12000, 1000)

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

    status.write("Step 1 Upload PDF to Blob")
    blob_upload_bytes(blob_service, BLOB_INPUT_CONTAINER, pdf_blob, pdf_bytes, "application/pdf")
    prog.progress(15)

    status.write("Step 2 Content Understanding")
    try:
        ir = content_understanding_ir(pdf_bytes)
    except Exception as ex:
        st.error(
            "Content Understanding failed.\n\n"
            f"Endpoint: {AZURE_CONTENT_UNDERSTANDING_ENDPOINT}\n"
            f"Analyzer ID: {CONTENT_UNDERSTANDING_ANALYZER_ID}\n\n"
            f"Error: {ex}"
        )
        st.stop()

    cu_conf = content_understanding_confidence(ir)
    blob_upload_json(blob_service, BLOB_LOG_CONTAINER, ir_blob, ir)
    prog.progress(45)

    status.write("Step 3 Azure OpenAI dynamic JSON")
    try:
        extracted = llm_dynamic_json(ir, user_hint=user_hint, chunk_chars=chunk_chars_extract)
    except Exception as ex:
        st.error(f"Dynamic JSON failed: {ex}")
        st.stop()

    extracted = normalize_llm_output(extracted)
    extracted.setdefault("payload", {})
    extracted.setdefault("evidence", {})
    extracted.setdefault("validations", {})

    extracted["payload"] = attach_field_confidence(
        extracted.get("payload"),
        evidence=extracted.get("evidence"),
        base_confidence=cu_conf["confidence_score"],
        ir=ir,
    )

    extracted["validations"]["content_understanding_confidence_components"] = cu_conf["components"]

    blob_upload_json(blob_service, BLOB_LOG_CONTAINER, extracted_blob, extracted)
    prog.progress(70)

    business_report = {}
    if run_business:
        status.write("Step 4 LLM business validation")
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

    status.write("Step 5 Deterministic validation")
    det_report = deterministic_validate(extracted)

    extracted["validations"]["business_validation"] = business_report
    extracted["validations"]["deterministic_validation"] = det_report

    extracted["metadata"] = extracted.get("metadata") or {}
    extracted["metadata"]["doc_id"] = doc_id
    extracted["metadata"]["created_utc"] = stamp
    extracted["metadata"]["source_blob"] = pdf_blob
    extracted["metadata"]["ir_blob"] = ir_blob
    extracted["metadata"]["cu_analyzer_id"] = CONTENT_UNDERSTANDING_ANALYZER_ID

    blob_upload_json(blob_service, BLOB_OUTPUT_CONTAINER, final_blob, extracted)
    prog.progress(100)
    status.write("Done")

    tab_words, tab_tables, tab_conf, tab_json = st.tabs(
        ["Words and confidence", "Tables", "CU confidence diagnostics", "Final JSON"]
    )

    with tab_words:
        st.subheader("Words with confidence scores")

        words_data = extract_word_confidence_from_ir(ir)
        stats = compute_word_confidence_stats(words_data)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total words", stats["total_words"])
        c2.metric("Avg confidence", "N/A" if stats["avg_confidence"] is None else f"{stats['avg_confidence']:.4f}")
        c3.metric("Low confidence under 0.8", stats["low_confidence_count"])

        min_confidence = st.slider("Minimum confidence score", 0.0, 1.0, 0.0, 0.05)
        filtered = [w for w in words_data if w.get("confidence") is not None and w["confidence"] >= min_confidence]

        if filtered:
            st.dataframe(pd.DataFrame(filtered), use_container_width=True, height=420)
            st.download_button(
                "Download words json",
                data=json.dumps(words_data, indent=2),
                file_name=f"{base}_words.json",
                mime="application/json",
            )
        else:
            st.info("No words matched the filter")

    with tab_tables:
        st.subheader("Extracted tables")
        tables = ir.get("tables") or []
        st.write("Tables detected:", len(tables))
        if tables:
            st.json(tables)
        else:
            st.info("No tables found")

    with tab_conf:
        st.subheader("Content Understanding confidence diagnostics")
        m1, m2, m3 = st.columns(3)
        m1.metric("OCR avg confidence", cu_conf["components"].get("ocr_avg_confidence"))
        m2.metric("Table cell OCR avg confidence", cu_conf["components"].get("table_cell_ocr_avg_confidence"))
        m3.metric("Composite document confidence_score", cu_conf.get("confidence_score"))
        st.json(cu_conf["components"])

    with tab_json:
        st.subheader("Final JSON with field and parent confidence_score")
        st.json(extracted)