"""
doc_qa.py

Ask questions directly against the extracted document JSON — no Azure AI Search needed.
Uses Azure OpenAI with the full extracted content as context.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


class DocumentQAError(Exception):
    pass


_DEFAULT_MAX_CHARS = 40_000   # ~10k tokens — safe for gpt-4o / gpt-4-turbo


def _extract_core_content(document_json: Dict[str, Any]) -> Any:
    for key in ("result", "analyzeResult", "contents", "extractedFields", "fields"):
        val = document_json.get(key)
        if val is not None:
            return val
    return document_json


def _build_flat_text(document_json: Dict[str, Any], max_chars: int) -> str:
    """Walk the JSON tree and pull out every text string with its location."""
    lines_out: List[str] = []

    def _walk_contents(contents: Any) -> None:
        if not isinstance(contents, list):
            return
        for content in contents:
            if not isinstance(content, dict):
                continue
            pages = content.get("pages") or []
            for page in pages:
                if not isinstance(page, dict):
                    continue
                pnum = page.get("pageNumber") or page.get("page") or "?"
                for line in (page.get("lines") or []):
                    if isinstance(line, dict):
                        txt = (line.get("content") or "").strip()
                        if txt:
                            lines_out.append(f"[P{pnum}] {txt}")

                for tidx, table in enumerate(page.get("tables") or []):
                    if not isinstance(table, dict):
                        continue
                    for cell in (table.get("cells") or []):
                        if not isinstance(cell, dict):
                            continue
                        txt = (cell.get("content") or "").strip()
                        r = cell.get("rowIndex", "?")
                        c = cell.get("columnIndex", "?")
                        if txt:
                            lines_out.append(f"[P{pnum} T{tidx} R{r}C{c}] {txt}")

    result = document_json.get("result") or document_json
    _walk_contents(result.get("contents"))

    flat = "\n".join(lines_out)
    if len(flat) > max_chars:
        flat = flat[:max_chars] + "\n...[truncated]"
    return flat


def build_document_context(
    document_json: Dict[str, Any],
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
    include_flat_text: bool = True,
) -> str:
    """Build a rich context string for the LLM."""
    parts: List[str] = []

    if include_flat_text:
        flat = _build_flat_text(document_json, max_chars=max_chars // 2)
        if flat.strip():
            parts.append("=== EXTRACTED TEXT (page / table / row / column) ===")
            parts.append(flat)
            parts.append("")

    core = _extract_core_content(document_json)
    json_text = json.dumps(core, indent=2, default=str)
    remaining = max_chars - sum(len(p) for p in parts)
    if len(json_text) > remaining:
        json_text = json_text[:remaining] + "\n...[truncated to fit model window]..."

    parts.append("=== RAW EXTRACTED JSON ===")
    parts.append(json_text)

    return "\n".join(parts)


def _derive_answer_confidence(
    *,
    conf_summary: Optional[Dict[str, Any]],
    grounded: bool,
    ocr_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Derive a user-facing answer confidence from:
    1. document extraction confidence
    2. whether the answer is grounded
    3. whether OCR warnings were needed

    This is not model probability.
    It is a practical confidence score for UI display.
    """
    mean_conf = (conf_summary or {}).get("mean_confidence")
    doc_quality = (conf_summary or {}).get("quality", "Unknown")
    ocr_warnings = ocr_warnings or []

    if mean_conf is None:
        base = 0.50
    else:
        base = float(mean_conf)

    if not grounded:
        base -= 0.25

    if ocr_warnings:
        base -= min(0.10, 0.02 * len(ocr_warnings))

    if doc_quality == "High":
        base += 0.05
    elif doc_quality == "Low":
        base -= 0.10
    elif doc_quality == "Unknown":
        base -= 0.05

    score = max(0.0, min(0.99, round(base, 3)))

    if score >= 0.85:
        label = "High"
    elif score >= 0.65:
        label = "Medium"
    else:
        label = "Low"

    return {
        "answer_confidence_score": score,
        "answer_confidence_label": label,
    }


_SYSTEM_PROMPT_STANDARD = """\
You are a precise financial and tax document analyst.

You have been given content extracted from a document by Azure Content Understanding.
The context contains two sections:
  1. EXTRACTED TEXT — a flat list of every line and table cell with page/row/column location.
  2. RAW EXTRACTED JSON — the full structured output.

Rules:
- Search BOTH sections thoroughly before concluding data is absent.
- Be specific: quote exact values and amounts where available.
- If the requested information is genuinely not present in either section, say so clearly
  and suggest what related data IS available.
- Never invent or estimate values.
- When multiple values match (e.g. multiple years), list all of them.

Return valid JSON only, no markdown fences:
{"answer": "<your detailed answer>", "grounded": true}

If the answer cannot be fully grounded, set "grounded": false and explain what is missing.
"""

_SYSTEM_PROMPT_LOW_CONFIDENCE = """\
You are an expert financial document analyst specialising in difficult, low-quality scans
and handwritten farm/agricultural financial documents.

The document was scanned and/or handwritten. OCR confidence is LOW, meaning:
- Some words may be misspelled or partially garbled (e.g. "beanng" = "bearing", "Jonagoid" = "Jonagold")
- Numbers may have digit errors (e.g. "8OOO" instead of "8000")
- Column alignment in tables may be off
- Field labels may be truncated or merged with adjacent values

The context contains two sections:
  1. EXTRACTED TEXT — a flat list of every line and table cell with page/row/column location.
     This is the MOST USEFUL section for handwritten documents. Read it carefully.
  2. RAW EXTRACTED JSON — the full structured output (may be harder to read for scanned docs).

Your job:
1. Search the EXTRACTED TEXT section first — look for lines that semantically match
   the question even if the spelling is imperfect.
2. Cross-reference with the RAW JSON to confirm values.
3. If OCR artefacts are present, interpret them intelligently and flag them
   (e.g. "OCR may have read 'beanng' as 'bearing'").
4. Always report what you DID find, even if it is only a partial or approximate match.
5. Never fabricate numbers. If a value is genuinely absent, say so AND list
   the related data that IS present so the user knows what was extracted.
6. For table data: use the R (row) and C (column) coordinates in the EXTRACTED TEXT
   to reconstruct rows even when the JSON structure is complex.

Return valid JSON only, no markdown fences:
{
  "answer": "<detailed answer with quoted values and page/location references>",
  "grounded": true,
  "ocr_warnings": ["list any suspected OCR errors you corrected or flagged"]
}

Set "grounded": false only if NO relevant data whatsoever was found.
"""


def ask_about_document(
    *,
    question: str,
    document_json: Dict[str, Any],
    file_name: str = "",
    conf_summary: Optional[Dict[str, Any]] = None,
    max_context_chars: int = _DEFAULT_MAX_CHARS,
    max_answer_tokens: int = 1500,
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    """
    Ask a question about the extracted document JSON.

    Returns:
        {
            "answer": str,
            "grounded": bool,
            "model_used": str,
            "ocr_warnings": list[str],
            "doc_quality": str,
            "answer_confidence_score": float,
            "answer_confidence_label": str,
        }
    """
    if not question or not question.strip():
        raise DocumentQAError("Question must not be empty.")
    if not document_json:
        raise DocumentQAError("No extracted document data available. Please analyze a document first.")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

    missing = [
        name for name, val in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ("AZURE_OPENAI_API_VERSION", api_version),
        ]
        if not val
    ]
    if missing:
        raise DocumentQAError(f"Missing environment variables: {', '.join(missing)}")

    quality = (conf_summary or {}).get("quality", "High")
    is_low_quality = quality in ("Low", "Medium", "Unknown")
    system_prompt = _SYSTEM_PROMPT_LOW_CONFIDENCE if is_low_quality else _SYSTEM_PROMPT_STANDARD

    context = build_document_context(
        document_json,
        max_chars=max_context_chars,
        include_flat_text=True,
    )

    parts = []
    if file_name:
        parts.append(f"Document file: {file_name}")

    if conf_summary:
        mean_c = conf_summary.get("mean_confidence")
        low_pct = conf_summary.get("low_conf_pct")
        if mean_c is not None:
            low_pct_text = f"{low_pct}%" if low_pct is not None else "N/A"
            parts.append(
                f"Document quality: {quality} "
                f"(mean confidence: {mean_c:.0%}, "
                f"{low_pct_text} of lines below 70% confidence)"
            )
        else:
            parts.append(f"Document quality: {quality}")

    parts.append("Extracted document content:")
    parts.append(context)
    parts.append(f"Question: {question.strip()}")
    user_message = "\n\n".join(parts)

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.0,
        "max_tokens": max_answer_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    except requests.exceptions.Timeout:
        raise DocumentQAError(f"Request timed out after {timeout_sec}s.")
    except Exception as exc:
        raise DocumentQAError(f"HTTP request failed: {exc}") from exc

    if resp.status_code != 200:
        raise DocumentQAError(f"Azure OpenAI HTTP {resp.status_code}: {resp.text[:600]}")

    choices: List[Dict[str, Any]] = resp.json().get("choices") or []
    if not choices:
        raise DocumentQAError("Azure OpenAI returned no choices.")

    raw_content: str = (choices[0].get("message") or {}).get("content", "")
    if not raw_content.strip():
        raise DocumentQAError("Azure OpenAI returned empty content.")

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        answer_conf = _derive_answer_confidence(
            conf_summary=conf_summary,
            grounded=False,
            ocr_warnings=[],
        )
        return {
            "answer": raw_content,
            "grounded": False,
            "model_used": deployment,
            "ocr_warnings": [],
            "doc_quality": quality,
            "answer_confidence_score": answer_conf["answer_confidence_score"],
            "answer_confidence_label": answer_conf["answer_confidence_label"],
        }

    ocr_warnings = parsed.get("ocr_warnings") or []
    grounded = bool(parsed.get("grounded", True))

    answer_conf = _derive_answer_confidence(
        conf_summary=conf_summary,
        grounded=grounded,
        ocr_warnings=ocr_warnings,
    )

    return {
        "answer": parsed.get("answer") or raw_content,
        "grounded": grounded,
        "model_used": deployment,
        "ocr_warnings": ocr_warnings,
        "doc_quality": quality,
        "answer_confidence_score": answer_conf["answer_confidence_score"],
        "answer_confidence_label": answer_conf["answer_confidence_label"],
    }