"""
doc_qa.py  (ui edition)

Ask questions directly against the extracted document JSON — no Azure AI Search needed.
Uses Azure OpenAI with the full extracted content as context.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

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


def build_document_context(document_json: Dict[str, Any], *, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    core = _extract_core_content(document_json)
    text = json.dumps(core, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...[truncated to fit model window]..."


_SYSTEM_PROMPT = """\
You are a precise financial and tax document analyst.

You have been given the full structured JSON output that was extracted from a document \
by Azure Content Understanding.

Rules:
- Answer ONLY from the data present in the provided JSON.
- Be specific: quote exact field names, values, and amounts where available.
- If the requested information is not present in the JSON, respond with:
  "This information was not found in the extracted document data."
- Never invent, estimate, or assume values.
- When multiple values match (e.g. multiple years), list all of them.

Return valid JSON only, no markdown fences:
{"answer": "<your detailed answer>", "grounded": true}

If the answer cannot be fully grounded, set "grounded": false and explain what is missing.
"""


def ask_about_document(
    *,
    question: str,
    document_json: Dict[str, Any],
    file_name: str = "",
    max_context_chars: int = _DEFAULT_MAX_CHARS,
    max_answer_tokens: int = 1200,
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    """
    Ask a question about the extracted document JSON.
    Returns {"answer": str, "grounded": bool, "model_used": str}
    """
    if not question or not question.strip():
        raise DocumentQAError("Question must not be empty.")
    if not document_json:
        raise DocumentQAError("No extracted document data available. Please analyze a document first.")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key  = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

    missing = [name for name, val in [
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", api_key),
        ("AZURE_OPENAI_DEPLOYMENT", deployment),
        ("AZURE_OPENAI_API_VERSION", api_version),
    ] if not val]
    if missing:
        raise DocumentQAError(f"Missing environment variables: {', '.join(missing)}")

    context = build_document_context(document_json, max_chars=max_context_chars)
    parts = []
    if file_name:
        parts.append(f"Document file: {file_name}")
    parts.append("Extracted document data (JSON):")
    parts.append(context)
    parts.append(f"Question: {question.strip()}")
    user_message = "\n\n".join(parts)

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": api_key}
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
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
        return {"answer": raw_content, "grounded": False, "model_used": deployment}

    return {
        "answer":     parsed.get("answer") or raw_content,
        "grounded":   bool(parsed.get("grounded", True)),
        "model_used": deployment,
    }
