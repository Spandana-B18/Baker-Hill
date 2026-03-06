"""
Azure Content Understanding client wrapper plus tax JSON generation.

This file does four things:

1. Calls Azure Content Understanding and polls for completion
2. Extracts words, lines, and table cells with confidence
3. Builds a compact evidence pack for tax extraction
4. Calls Azure OpenAI to generate final tax JSON in envelope form
   with field confidence_score and parent confidence_score
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


class ContentUnderstandingError(Exception):
    pass


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class Span:
    offset: int
    length: int

    @property
    def end(self) -> int:
        return self.offset + self.length


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _overlaps(a: Span, b: Span) -> bool:
    return not (a.end <= b.offset or b.end <= a.offset)


def _extract_spans(node: Dict[str, Any]) -> List[Span]:
    spans_raw = node.get("spans")
    if isinstance(spans_raw, list) and spans_raw:
        spans: List[Span] = []
        for s in spans_raw:
            if isinstance(s, dict):
                span = Span(
                    offset=_safe_int(s.get("offset"), 0),
                    length=_safe_int(s.get("length"), 0),
                )
                if span.length > 0:
                    spans.append(span)
        return spans

    span_raw = node.get("span")
    if isinstance(span_raw, dict):
        span = Span(
            offset=_safe_int(span_raw.get("offset"), 0),
            length=_safe_int(span_raw.get("length"), 0),
        )
        return [span] if span.length > 0 else []

    return []


class ContentUnderstandingClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        *,
        poll_interval_sec: float = 2.0,
        max_poll_seconds: float = 300.0,
        session: Optional[requests.Session] = None,
    ):
        if not endpoint:
            raise ContentUnderstandingError("Missing endpoint")
        if not api_key:
            raise ContentUnderstandingError("Missing api key")
        if not api_version:
            raise ContentUnderstandingError("Missing api version")

        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.poll_interval_sec = float(poll_interval_sec)
        self.max_poll_seconds = float(max_poll_seconds)
        self._session = session or requests.Session()
        self._headers = {"Ocp-Apim-Subscription-Key": api_key}

    def analyze_document(
        self,
        *,
        analyzer_id: str,
        file_bytes: bytes,
        file_name: str,
        content_type: str = "application/octet-stream",
    ) -> Dict[str, Any]:
        operation_url, immediate = self._submit(
            analyzer_id=analyzer_id,
            file_bytes=file_bytes,
            file_name=file_name,
            content_type=content_type,
        )
        if immediate is not None:
            return immediate
        return self._poll(operation_url)

    def _submit(
        self,
        *,
        analyzer_id: str,
        file_bytes: bytes,
        file_name: str,
        content_type: str,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if not analyzer_id:
            raise ContentUnderstandingError("Missing analyzer id")
        if not file_bytes:
            raise ContentUnderstandingError("Empty file bytes")

        url = (
            f"{self.endpoint}/contentunderstanding/analyzers/"
            f"{analyzer_id}:analyze?api-version={self.api_version}"
        )

        headers = dict(self._headers)
        headers["Content-Type"] = content_type

        response = self._session.post(url, headers=headers, data=file_bytes, timeout=120)

        if response.status_code not in (200, 202):
            raise ContentUnderstandingError(
                f"Submission failed {response.status_code}: {response.text}"
            )

        operation_url = response.headers.get("Operation-Location") or response.headers.get(
            "operation-location"
        )
        if operation_url:
            return operation_url, None

        if response.status_code == 200:
            return None, response.json()

        raise ContentUnderstandingError("Missing Operation-Location header")

    def _poll(self, operation_url: str) -> Dict[str, Any]:
        deadline = time.time() + self.max_poll_seconds

        while True:
            if time.time() > deadline:
                raise ContentUnderstandingError("Timed out waiting for analysis result")

            response = self._session.get(operation_url, headers=self._headers, timeout=60)

            if response.status_code != 200:
                raise ContentUnderstandingError(
                    f"Polling failed {response.status_code}: {response.text}"
                )

            data = response.json()
            status = str(data.get("status", "")).lower()

            if status == "succeeded":
                return data

            if status in ("failed", "canceled"):
                error = data.get("error") or {}
                message = error.get("message") or "unknown error"
                raise ContentUnderstandingError(f"Analysis {status}: {message}")

            time.sleep(self.poll_interval_sec)

    @staticmethod
    def _root_result(result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        return result.get("result") if isinstance(result.get("result"), dict) else result

    @staticmethod
    def iter_pages(result: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        root = ContentUnderstandingClient._root_result(result)
        contents = root.get("contents")
        if not isinstance(contents, list):
            return

        for content in contents:
            if not isinstance(content, dict):
                continue
            pages = content.get("pages")
            if not isinstance(pages, list):
                continue
            for page in pages:
                if isinstance(page, dict):
                    yield page

    @staticmethod
    def extract_words(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []

        for page in ContentUnderstandingClient.iter_pages(result):
            page_number = _safe_int(_coalesce(page.get("pageNumber"), page.get("page")), 0)
            words = page.get("words") or []
            if not isinstance(words, list):
                continue

            for idx, word in enumerate(words):
                if not isinstance(word, dict):
                    continue

                spans = _extract_spans(word)
                span = spans[0] if spans else Span(0, 0)

                rows.append(
                    {
                        "page": page_number,
                        "word_index": idx,
                        "text": word.get("content", ""),
                        "confidence": _safe_float(word.get("confidence")),
                        "span": {
                            "offset": span.offset,
                            "length": span.length,
                        },
                        "source": word.get("source"),
                    }
                )

        return rows

    @staticmethod
    def _words_index_by_page(result: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
        by_page: Dict[int, List[Dict[str, Any]]] = {}

        for row in ContentUnderstandingClient.extract_words(result):
            page = _safe_int(row.get("page"), 0)
            by_page.setdefault(page, []).append(row)

        for page, rows in by_page.items():
            rows.sort(key=lambda x: _safe_int(x.get("span", {}).get("offset"), 0))
            by_page[page] = rows

        return by_page

    @staticmethod
    def _aggregate_confidence_for_spans(
        *,
        words: List[Dict[str, Any]],
        spans: List[Span],
        mode: str = "mean",
    ) -> Optional[float]:
        if not spans:
            return None

        confidences: List[float] = []

        for word in words:
            confidence = _safe_float(word.get("confidence"))
            if confidence is None:
                continue

            span_info = word.get("span", {})
            word_span = Span(
                offset=_safe_int(span_info.get("offset"), 0),
                length=_safe_int(span_info.get("length"), 0),
            )
            if word_span.length <= 0:
                continue

            for span in spans:
                if _overlaps(word_span, span):
                    confidences.append(confidence)
                    break

        if not confidences:
            return None

        if mode == "min":
            return min(confidences)

        return _mean(confidences)

    @staticmethod
    def extract_lines_with_confidence(
        result: Dict[str, Any],
        *,
        aggregate_mode: str = "mean",
        max_lines_per_page: int = 250,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        words_by_page = ContentUnderstandingClient._words_index_by_page(result)

        for page in ContentUnderstandingClient.iter_pages(result):
            page_number = _safe_int(_coalesce(page.get("pageNumber"), page.get("page")), 0)
            page_words = words_by_page.get(page_number, [])

            lines = page.get("lines") or []
            if not isinstance(lines, list):
                continue

            for idx, line in enumerate(lines[:max_lines_per_page]):
                if not isinstance(line, dict):
                    continue

                spans = _extract_spans(line)
                confidence = ContentUnderstandingClient._aggregate_confidence_for_spans(
                    words=page_words,
                    spans=spans,
                    mode=aggregate_mode,
                )

                rows.append(
                    {
                        "page": page_number,
                        "line_index": idx,
                        "text": line.get("content", ""),
                        "confidence": confidence,
                        "spans": [{"offset": s.offset, "length": s.length} for s in spans],
                        "source": line.get("source"),
                    }
                )

        return rows

    @staticmethod
    def extract_table_cells_with_confidence(
        result: Dict[str, Any],
        *,
        aggregate_mode: str = "mean",
        max_tables_per_page: int = 50,
        max_cells_per_table: int = 500,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        words_by_page = ContentUnderstandingClient._words_index_by_page(result)

        for page in ContentUnderstandingClient.iter_pages(result):
            page_number = _safe_int(_coalesce(page.get("pageNumber"), page.get("page")), 0)
            page_words = words_by_page.get(page_number, [])

            tables = page.get("tables") or []
            if not isinstance(tables, list):
                continue

            for table_idx, table in enumerate(tables[:max_tables_per_page]):
                if not isinstance(table, dict):
                    continue

                cells = table.get("cells") or []
                if not isinstance(cells, list):
                    continue

                for cell_idx, cell in enumerate(cells[:max_cells_per_table]):
                    if not isinstance(cell, dict):
                        continue

                    spans = _extract_spans(cell)
                    confidence = ContentUnderstandingClient._aggregate_confidence_for_spans(
                        words=page_words,
                        spans=spans,
                        mode=aggregate_mode,
                    )

                    rows.append(
                        {
                            "page": page_number,
                            "table_index": table_idx,
                            "cell_index": cell_idx,
                            "row_index": _safe_int(cell.get("rowIndex"), 0),
                            "column_index": _safe_int(cell.get("columnIndex"), 0),
                            "kind": cell.get("kind"),
                            "text": cell.get("content", ""),
                            "confidence": confidence,
                            "spans": [{"offset": s.offset, "length": s.length} for s in spans],
                        }
                    )

        return rows


def default_analyzer_id() -> str:
    return "prebuilt-layout"


def _azure_openai_chat_completion(
    *,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 2500,
) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = os.getenv("AZURE_OPENAI_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

    if not endpoint or not key or not deployment or not api_version:
        raise LLMError("Missing Azure OpenAI configuration in environment variables")

    url = (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={api_version}"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": key,
    }

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    timeout_sec = _safe_int(os.getenv("LLM_TIMEOUT_SEC", "90"), 90)

    response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)

    if response.status_code != 200:
        raise LLMError(f"Azure OpenAI call failed {response.status_code}: {response.text}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("No choices returned by Azure OpenAI")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMError("Empty LLM response content")

    return content.strip()


def build_evidence_pack_for_tax(raw_result: Dict[str, Any], max_chars: int = 30000) -> Dict[str, Any]:
    """
    Build a compact evidence pack for the LLM from high confidence lines and table cells.
    """
    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    cells = ContentUnderstandingClient.extract_table_cells_with_confidence(raw_result, aggregate_mode="mean")

    def sort_key(item: Dict[str, Any]) -> float:
        confidence = _safe_float(item.get("confidence"))
        return confidence if confidence is not None else 0.0

    lines = sorted(lines, key=sort_key, reverse=True)
    cells = sorted(cells, key=sort_key, reverse=True)

    pack = {
        "lines": [
            {
                "page": row.get("page"),
                "text": row.get("text"),
                "confidence": row.get("confidence"),
            }
            for row in lines[:1500]
            if (row.get("text") or "").strip()
        ],
        "table_cells": [
            {
                "page": row.get("page"),
                "table_index": row.get("table_index"),
                "row_index": row.get("row_index"),
                "column_index": row.get("column_index"),
                "text": row.get("text"),
                "confidence": row.get("confidence"),
            }
            for row in cells[:1500]
            if (row.get("text") or "").strip()
        ],
    }

    raw = json.dumps(pack, ensure_ascii=False)
    if len(raw) <= max_chars:
        return pack

    ratio = max_chars / max(1, len(raw))
    keep_lines = max(80, int(len(pack["lines"]) * ratio))
    keep_cells = max(80, int(len(pack["table_cells"]) * ratio))

    return {
        "lines": pack["lines"][:keep_lines],
        "table_cells": pack["table_cells"][:keep_cells],
    }


def _normalize_leaf_node(node: Any) -> Dict[str, Any]:
    if isinstance(node, dict) and "value" in node and "confidence_score" in node:
        confidence = _safe_float(node.get("confidence_score"))
        return {
            "value": node.get("value"),
            "confidence_score": 0.0 if confidence is None else round(confidence, 3),
        }

    return {
        "value": node,
        "confidence_score": 0.0,
    }


def _collect_child_confidences(node: Any) -> List[float]:
    confidences: List[float] = []

    if isinstance(node, dict):
        if "confidence_score" in node and isinstance(node.get("confidence_score"), (int, float)):
            confidences.append(float(node["confidence_score"]))

        for key, value in node.items():
            if key == "confidence_score":
                continue
            confidences.extend(_collect_child_confidences(value))

    elif isinstance(node, list):
        for item in node:
            confidences.extend(_collect_child_confidences(item))

    return confidences


def _add_parent_confidence(node: Any) -> Any:
    """
    Recursively add confidence_score to parent dictionaries.
    Leaf shape is preserved as
    {
      "value": ...,
      "confidence_score": ...
    }
    """
    if isinstance(node, dict):
        for key, value in list(node.items()):
            node[key] = _add_parent_confidence(value)

        is_leaf = set(node.keys()) == {"value", "confidence_score"}

        if not is_leaf:
            child_confidences: List[float] = []
            for key, value in node.items():
                if key == "confidence_score":
                    continue
                child_confidences.extend(_collect_child_confidences(value))

            node["confidence_score"] = round(_mean(child_confidences) or 0.0, 3)

        return node

    if isinstance(node, list):
        return [_add_parent_confidence(item) for item in node]

    return node


def _make_tax_1120s_prompt(
    evidence: Dict[str, Any],
    *,
    doc_id: str,
    created_utc: str,
    source_blob: str,
    ir_blob: str,
    cu_analyzer_id: str,
) -> List[Dict[str, str]]:
    system = """
You are an extraction engine for IRS tax documents.

Return only valid JSON.
Do not use markdown.
Do not invent values.
Use only the evidence provided.
If a value cannot be found, set its value to null and confidence_score to 0.

Required output shape:

{
  "metadata": {
    "chunking": {
      "chunks": <number>
    },
    "doc_id": "<string>",
    "created_utc": "<string>",
    "source_blob": "<string>",
    "ir_blob": "<string>",
    "cu_analyzer_id": "<string>"
  },
  "schema": {
    "schema_id": "tax_1120s",
    "schema_version": "2024"
  },
  "payload": {
    "tax_year": {
      "value": <string or null>,
      "confidence_score": <number>
    },
    "corporation_name": {
      "value": <string or null>,
      "confidence_score": <number>
    },
    "employer_identification_number": {
      "value": <string or null>,
      "confidence_score": <number>
    },
    "business_activity_code": {
      "value": <string or null>,
      "confidence_score": <number>
    },
    "address": {
      "street": {
        "value": <string or null>,
        "confidence_score": <number>
      },
      "city": {
        "value": <string or null>,
        "confidence_score": <number>
      },
      "state": {
        "value": <string or null>,
        "confidence_score": <number>
      },
      "zip_code": {
        "value": <string or null>,
        "confidence_score": <number>
      }
    },
    "ordinary_business_income_loss": {
      "value": <number or string or null>,
      "confidence_score": <number>
    },
    "total_assets_end_of_year": {
      "value": <number or string or null>,
      "confidence_score": <number>
    },
    "cash": {
      "value": <number or string or null>,
      "confidence_score": <number>
    },
    "accounts_receivable": {
      "value": <number or string or null>,
      "confidence_score": <number>
    }
  }
}

Important:
Only populate what is supported by evidence.
The payload must always be an object.
Do not add explanations.
"""

    user = {
        "metadata_values": {
            "doc_id": doc_id,
            "created_utc": created_utc,
            "source_blob": source_blob,
            "ir_blob": ir_blob,
            "cu_analyzer_id": cu_analyzer_id,
        },
        "chunk_count": len(evidence.get("lines", [])) + len(evidence.get("table_cells", [])),
        "evidence": evidence,
    }

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _enforce_tax_payload_shape(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Force the payload into the exact leaf style:
    each field is { value, confidence_score }
    and parents later get confidence_score recursively.
    """
    payload = payload if isinstance(payload, dict) else {}

    def get_leaf(key: str) -> Dict[str, Any]:
        return _normalize_leaf_node(payload.get(key, {"value": None, "confidence_score": 0.0}))

    address_raw = payload.get("address", {})
    address_raw = address_raw if isinstance(address_raw, dict) else {}

    address = {
        "street": _normalize_leaf_node(address_raw.get("street", {"value": None, "confidence_score": 0.0})),
        "city": _normalize_leaf_node(address_raw.get("city", {"value": None, "confidence_score": 0.0})),
        "state": _normalize_leaf_node(address_raw.get("state", {"value": None, "confidence_score": 0.0})),
        "zip_code": _normalize_leaf_node(address_raw.get("zip_code", {"value": None, "confidence_score": 0.0})),
    }

    normalized = {
        "tax_year": get_leaf("tax_year"),
        "corporation_name": get_leaf("corporation_name"),
        "employer_identification_number": get_leaf("employer_identification_number"),
        "business_activity_code": get_leaf("business_activity_code"),
        "address": address,
        "ordinary_business_income_loss": get_leaf("ordinary_business_income_loss"),
        "total_assets_end_of_year": get_leaf("total_assets_end_of_year"),
        "cash": get_leaf("cash"),
        "accounts_receivable": get_leaf("accounts_receivable"),
    }

    return _add_parent_confidence(normalized)


def generate_tax_1120s_envelope(
    raw_result: Dict[str, Any],
    *,
    doc_id: str,
    created_utc: str,
    source_blob: str,
    ir_blob: str,
    cu_analyzer_id: str = "prebuilt-layout",
    max_evidence_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate final JSON in this form:

    {
      "metadata": {...},
      "schema": {...},
      "payload": {
        "field_name": {
          "value": ...,
          "confidence_score": ...
        },
        "parent_object": {
          ...
          "confidence_score": ...
        }
      }
    }
    """
    max_chars = max_evidence_chars or _safe_int(os.getenv("LLM_MAX_EVIDENCE_CHARS", "30000"), 30000)
    evidence = build_evidence_pack_for_tax(raw_result, max_chars=max_chars)

    messages = _make_tax_1120s_prompt(
        evidence,
        doc_id=doc_id,
        created_utc=created_utc,
        source_blob=source_blob,
        ir_blob=ir_blob,
        cu_analyzer_id=cu_analyzer_id,
    )

    text = _azure_openai_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=2500,
    )

    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise LLMError(f"LLM did not return valid JSON: {exc}")

    if not isinstance(parsed, dict):
        raise LLMError("LLM root must be a JSON object")

    raw_payload = parsed.get("payload", {})
    payload = _enforce_tax_payload_shape(raw_payload)

    chunk_count = 0
    root = raw_result.get("result")
    if isinstance(root, dict):
        contents = root.get("contents")
        if isinstance(contents, list):
            chunk_count = len(contents)

    final_json = {
        "metadata": {
            "chunking": {
                "chunks": chunk_count,
            },
            "doc_id": doc_id,
            "created_utc": created_utc,
            "source_blob": source_blob,
            "ir_blob": ir_blob,
            "cu_analyzer_id": cu_analyzer_id,
        },
        "schema": {
            "schema_id": "tax_1120s",
            "schema_version": "2024",
        },
        "payload": payload,
    }

    final_json["payload"] = _add_parent_confidence(final_json["payload"])
    return final_json