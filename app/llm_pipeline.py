from services.azure_openai import azure_openai_client
from utils.ir_processing import build_table_snippet, split_into_chunks, merge_envelopes
from schema.registry import SCHEMA_REGISTRY
from schema.envelope import ENVELOPE_SCHEMA
import json
from config import AZURE_OPENAI_DEPLOYMENT

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