"""
conf_score.py

Production ready module for:

1. Calling Azure Content Understanding
2. Extracting words, lines, table cells, and paragraphs with confidence
3. Detecting document type dynamically
4. Building a dynamic envelope for tax documents, financial documents, or generic documents
5. Using Azure OpenAI to structure tax and financial payloads
6. Adding confidence_score to leaves and parent objects

Expected environment variables for Azure OpenAI:
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_KEY
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_VERSION

Optional:
LLM_TIMEOUT_SEC
LLM_MAX_EVIDENCE_CHARS
"""

from __future__ import annotations

import json
import os
import re
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


def default_analyzer_id() -> str:
    return "prebuilt-layout"


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
        max_lines_per_page: int = 300,
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
    def extract_paragraphs_with_confidence(
        result: Dict[str, Any],
        *,
        aggregate_mode: str = "mean",
        max_paragraphs_per_page: int = 250,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        words_by_page = ContentUnderstandingClient._words_index_by_page(result)

        for page in ContentUnderstandingClient.iter_pages(result):
            page_number = _safe_int(_coalesce(page.get("pageNumber"), page.get("page")), 0)
            page_words = words_by_page.get(page_number, [])

            paragraphs = page.get("paragraphs") or []
            if not isinstance(paragraphs, list):
                continue

            for idx, paragraph in enumerate(paragraphs[:max_paragraphs_per_page]):
                if not isinstance(paragraph, dict):
                    continue

                spans = _extract_spans(paragraph)
                confidence = ContentUnderstandingClient._aggregate_confidence_for_spans(
                    words=page_words,
                    spans=spans,
                    mode=aggregate_mode,
                )

                rows.append(
                    {
                        "page": page_number,
                        "paragraph_index": idx,
                        "text": paragraph.get("content", ""),
                        "confidence": confidence,
                        "role": paragraph.get("role"),
                        "spans": [{"offset": s.offset, "length": s.length} for s in spans],
                        "source": paragraph.get("source"),
                    }
                )

        return rows

    @staticmethod
    def extract_table_cells_with_confidence(
        result: Dict[str, Any],
        *,
        aggregate_mode: str = "mean",
        max_tables_per_page: int = 50,
        max_cells_per_table: int = 800,
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


def _azure_openai_chat_completion(
    *,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 3500,
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

    timeout_sec = _safe_int(os.getenv("LLM_TIMEOUT_SEC", "120"), 120)

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


def _compact_lines(lines: List[Dict[str, Any]], limit: int = 400) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in lines[:limit]:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        compact.append(
            {
                "page": row.get("page"),
                "text": text,
                "confidence": row.get("confidence"),
            }
        )
    return compact


def _compact_paragraphs(paragraphs: List[Dict[str, Any]], limit: int = 250) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in paragraphs[:limit]:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        compact.append(
            {
                "page": row.get("page"),
                "paragraph_index": row.get("paragraph_index"),
                "role": row.get("role"),
                "text": text,
                "confidence": row.get("confidence"),
            }
        )
    return compact


def _compact_table_cells(cells: List[Dict[str, Any]], limit: int = 1200) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in cells[:limit]:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        compact.append(
            {
                "page": row.get("page"),
                "table_index": row.get("table_index"),
                "row_index": row.get("row_index"),
                "column_index": row.get("column_index"),
                "kind": row.get("kind"),
                "text": text,
                "confidence": row.get("confidence"),
            }
        )
    return compact


def build_evidence_pack(
    raw_result: Dict[str, Any],
    *,
    max_chars: int = 35000,
) -> Dict[str, Any]:
    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(raw_result, aggregate_mode="mean")
    table_cells = ContentUnderstandingClient.extract_table_cells_with_confidence(raw_result, aggregate_mode="mean")

    def sort_key(item: Dict[str, Any]) -> float:
        confidence = _safe_float(item.get("confidence"))
        return confidence if confidence is not None else 0.0

    lines_sorted = sorted(lines, key=sort_key, reverse=True)
    paragraphs_sorted = sorted(paragraphs, key=sort_key, reverse=True)
    table_cells_sorted = sorted(table_cells, key=sort_key, reverse=True)

    pack = {
        "lines": _compact_lines(lines_sorted, limit=500),
        "paragraphs": _compact_paragraphs(paragraphs_sorted, limit=250),
        "table_cells": _compact_table_cells(table_cells_sorted, limit=1500),
    }

    raw = json.dumps(pack, ensure_ascii=False)
    if len(raw) <= max_chars:
        return pack

    ratio = max_chars / max(1, len(raw))
    keep_lines = max(80, int(len(pack["lines"]) * ratio))
    keep_paragraphs = max(40, int(len(pack["paragraphs"]) * ratio))
    keep_cells = max(150, int(len(pack["table_cells"]) * ratio))

    return {
        "lines": pack["lines"][:keep_lines],
        "paragraphs": pack["paragraphs"][:keep_paragraphs],
        "table_cells": pack["table_cells"][:keep_cells],
    }


def detect_document_schema(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic router for document family and schema.
    """
    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(raw_result, aggregate_mode="mean")

    samples: List[str] = []
    for row in lines[:120]:
        text = (row.get("text") or "").strip()
        if text:
            samples.append(text.lower())

    for row in paragraphs[:40]:
        text = (row.get("text") or "").strip()
        if text:
            samples.append(text.lower())

    joined = "\n".join(samples)

    tax_patterns = {
        "1120s": [
            "form 1120-s",
            "1120-s",
            "u.s. income tax return for an s corporation",
            "s corporation",
            "schedule l",
            "ordinary business income",
            "employer identification number",
            "internal revenue service",
        ],
        "k1": [
            "schedule k-1",
            "shareholder's share",
            "partner's share",
            "form 1065",
            "form 1120s schedule k-1",
        ],
        "1040": [
            "form 1040",
            "u.s. individual income tax return",
            "filing status",
            "dependents",
            "adjusted gross income",
        ],
    }

    financial_patterns = {
        "financial_statement": [
            "balance sheet",
            "statement of financial position",
            "income statement",
            "statement of cash flows",
            "statement of owner equity",
            "statement of owners equity",
            "ratio analysis",
            "current assets",
            "current liabilities",
            "retained earnings",
            "net income",
            "total assets",
            "total liabilities",
            "cash and cash equivalents",
        ]
    }

    scores: Dict[str, float] = {
        "tax_1120s": 0.0,
        "tax_k1": 0.0,
        "tax_1040": 0.0,
        "financial_statement": 0.0,
        "generic_document": 0.0,
    }

    for p in tax_patterns["1120s"]:
        if p in joined:
            scores["tax_1120s"] += 1.0

    for p in tax_patterns["k1"]:
        if p in joined:
            scores["tax_k1"] += 1.0

    for p in tax_patterns["1040"]:
        if p in joined:
            scores["tax_1040"] += 1.0

    for p in financial_patterns["financial_statement"]:
        if p in joined:
            scores["financial_statement"] += 1.0

    best_schema = max(scores, key=scores.get)
    best_score = scores[best_schema]

    if best_score <= 0:
        return {
            "document_type": "generic_document",
            "document_subtype": "generic",
            "schema_id": "generic_document",
            "schema_version": "2024",
            "confidence_score": 0.2,
        }

    if best_schema.startswith("tax_"):
        subtype = best_schema.replace("tax_", "")
        return {
            "document_type": "tax_document",
            "document_subtype": subtype,
            "schema_id": best_schema,
            "schema_version": "2024",
            "confidence_score": round(min(0.99, 0.55 + (0.06 * best_score)), 3),
        }

    if best_schema == "financial_statement":
        return {
            "document_type": "financial_document",
            "document_subtype": "statement",
            "schema_id": "financial_statement",
            "schema_version": "2024",
            "confidence_score": round(min(0.99, 0.55 + (0.05 * best_score)), 3),
        }

    return {
        "document_type": "generic_document",
        "document_subtype": "generic",
        "schema_id": "generic_document",
        "schema_version": "2024",
        "confidence_score": 0.2,
    }


def _extract_document_title(raw_result: Dict[str, Any]) -> Optional[str]:
    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(raw_result, aggregate_mode="mean")
    if paragraphs:
        first_page = [p for p in paragraphs if p.get("page") == 1 and (p.get("text") or "").strip()]
        if first_page:
            first_page.sort(key=lambda x: (x.get("paragraph_index", 999999), -(x.get("confidence") or 0.0)))
            return first_page[0].get("text")

    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    first_page_lines = [l for l in lines if l.get("page") == 1 and (l.get("text") or "").strip()]
    if first_page_lines:
        first_page_lines.sort(key=lambda x: x.get("line_index", 999999))
        return first_page_lines[0].get("text")

    return None


def _llm_json(messages: List[Dict[str, Any]], *, max_tokens: int = 3500) -> Dict[str, Any]:
    text = _azure_openai_chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens)
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise LLMError(f"LLM did not return valid JSON: {exc}")

    if not isinstance(parsed, dict):
        raise LLMError("LLM root must be a JSON object")

    return parsed


def _make_tax_prompt(
    evidence: Dict[str, Any],
    *,
    schema_id: str,
) -> List[Dict[str, Any]]:
    system = """
You are a tax document extraction engine.

Return only valid JSON.
Do not use markdown.
Do not invent values.
Use only the evidence provided.
If a value cannot be found, set its value to null and confidence_score to 0.

You must return a JSON object with this structure:

{
  "payload": {
    "form_type": {"value": <string or null>, "confidence_score": <number>},
    "tax_year": {"value": <string or null>, "confidence_score": <number>},
    "entity_name": {"value": <string or null>, "confidence_score": <number>},
    "employer_identification_number": {"value": <string or null>, "confidence_score": <number>},
    "business_activity_code": {"value": <string or null>, "confidence_score": <number>},
    "address": {
      "street": {"value": <string or null>, "confidence_score": <number>},
      "city": {"value": <string or null>, "confidence_score": <number>},
      "state": {"value": <string or null>, "confidence_score": <number>},
      "zip_code": {"value": <string or null>, "confidence_score": <number>}
    },
    "key_amounts": {
      "ordinary_business_income_loss": {"value": <number or string or null>, "confidence_score": <number>},
      "total_assets_end_of_year": {"value": <number or string or null>, "confidence_score": <number>},
      "cash": {"value": <number or string or null>, "confidence_score": <number>},
      "accounts_receivable": {"value": <number or string or null>, "confidence_score": <number>}
    }
  }
}

The payload must always be an object.
Only fill fields supported by evidence.
"""
    user = {
        "schema_id": schema_id,
        "evidence": evidence,
    }
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _make_financial_prompt(
    evidence: Dict[str, Any],
    *,
    schema_id: str,
) -> List[Dict[str, Any]]:
    system = """
You are a financial document extraction engine.

Return only valid JSON.
Do not use markdown.
Do not invent values.
Use only the evidence provided.
If a value cannot be found, set its value to null and confidence_score to 0.

You must return a JSON object with this structure:

{
  "payload": {
    "statements_included": [
      {"value": <string or null>, "confidence_score": <number>}
    ],
    "balance_sheet": {
      "current_assets": {
        "cash_and_cash_equivalents": {"value": <number or string or null>, "confidence_score": <number>},
        "accounts_receivable": {"value": <number or string or null>, "confidence_score": <number>},
        "inventory": {"value": <number or string or null>, "confidence_score": <number>}
      },
      "non_current_assets": {
        "property_plant_equipment": {"value": <number or string or null>, "confidence_score": <number>}
      },
      "current_liabilities": {
        "accounts_payable": {"value": <number or string or null>, "confidence_score": <number>}
      },
      "equity": {
        "owner_equity": {"value": <number or string or null>, "confidence_score": <number>},
        "retained_earnings": {"value": <number or string or null>, "confidence_score": <number>}
      },
      "totals": {
        "total_assets": {"value": <number or string or null>, "confidence_score": <number>},
        "total_liabilities": {"value": <number or string or null>, "confidence_score": <number>},
        "total_equity": {"value": <number or string or null>, "confidence_score": <number>}
      }
    },
    "income_statement": {
      "revenue": {"value": <number or string or null>, "confidence_score": <number>},
      "cost_of_goods_sold": {"value": <number or string or null>, "confidence_score": <number>},
      "operating_expenses": {"value": <number or string or null>, "confidence_score": <number>},
      "net_income": {"value": <number or string or null>, "confidence_score": <number>}
    },
    "cash_flow_statement": {
      "net_cash_from_operations": {"value": <number or string or null>, "confidence_score": <number>},
      "net_cash_from_investing": {"value": <number or string or null>, "confidence_score": <number>},
      "net_cash_from_financing": {"value": <number or string or null>, "confidence_score": <number>}
    }
  }
}

The payload must always be an object.
Only fill fields supported by evidence.
"""
    user = {
        "schema_id": schema_id,
        "evidence": evidence,
    }
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _normalize_tax_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}

    def leaf_at(container: Dict[str, Any], key: str) -> Dict[str, Any]:
        return _normalize_leaf_node(container.get(key, {"value": None, "confidence_score": 0.0}))

    address_raw = payload.get("address", {})
    address_raw = address_raw if isinstance(address_raw, dict) else {}

    key_amounts_raw = payload.get("key_amounts", {})
    key_amounts_raw = key_amounts_raw if isinstance(key_amounts_raw, dict) else {}

    normalized = {
        "form_type": leaf_at(payload, "form_type"),
        "tax_year": leaf_at(payload, "tax_year"),
        "entity_name": leaf_at(payload, "entity_name"),
        "employer_identification_number": leaf_at(payload, "employer_identification_number"),
        "business_activity_code": leaf_at(payload, "business_activity_code"),
        "address": {
            "street": _normalize_leaf_node(address_raw.get("street", {"value": None, "confidence_score": 0.0})),
            "city": _normalize_leaf_node(address_raw.get("city", {"value": None, "confidence_score": 0.0})),
            "state": _normalize_leaf_node(address_raw.get("state", {"value": None, "confidence_score": 0.0})),
            "zip_code": _normalize_leaf_node(address_raw.get("zip_code", {"value": None, "confidence_score": 0.0})),
        },
        "key_amounts": {
            "ordinary_business_income_loss": _normalize_leaf_node(
                key_amounts_raw.get("ordinary_business_income_loss", {"value": None, "confidence_score": 0.0})
            ),
            "total_assets_end_of_year": _normalize_leaf_node(
                key_amounts_raw.get("total_assets_end_of_year", {"value": None, "confidence_score": 0.0})
            ),
            "cash": _normalize_leaf_node(
                key_amounts_raw.get("cash", {"value": None, "confidence_score": 0.0})
            ),
            "accounts_receivable": _normalize_leaf_node(
                key_amounts_raw.get("accounts_receivable", {"value": None, "confidence_score": 0.0})
            ),
        },
    }

    return _add_parent_confidence(normalized)


def _normalize_financial_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}

    def leaf_from(container: Dict[str, Any], key: str) -> Dict[str, Any]:
        return _normalize_leaf_node(container.get(key, {"value": None, "confidence_score": 0.0}))

    statements_raw = payload.get("statements_included", [])
    statements_included: List[Dict[str, Any]] = []
    if isinstance(statements_raw, list):
        for item in statements_raw:
            statements_included.append(_normalize_leaf_node(item))
    else:
        statements_included = []

    bs_raw = payload.get("balance_sheet", {})
    bs_raw = bs_raw if isinstance(bs_raw, dict) else {}
    bs_current_assets = bs_raw.get("current_assets", {})
    bs_current_assets = bs_current_assets if isinstance(bs_current_assets, dict) else {}
    bs_non_current_assets = bs_raw.get("non_current_assets", {})
    bs_non_current_assets = bs_non_current_assets if isinstance(bs_non_current_assets, dict) else {}
    bs_current_liabilities = bs_raw.get("current_liabilities", {})
    bs_current_liabilities = bs_current_liabilities if isinstance(bs_current_liabilities, dict) else {}
    bs_equity = bs_raw.get("equity", {})
    bs_equity = bs_equity if isinstance(bs_equity, dict) else {}
    bs_totals = bs_raw.get("totals", {})
    bs_totals = bs_totals if isinstance(bs_totals, dict) else {}

    is_raw = payload.get("income_statement", {})
    is_raw = is_raw if isinstance(is_raw, dict) else {}

    cf_raw = payload.get("cash_flow_statement", {})
    cf_raw = cf_raw if isinstance(cf_raw, dict) else {}

    normalized = {
        "statements_included": statements_included,
        "balance_sheet": {
            "current_assets": {
                "cash_and_cash_equivalents": _normalize_leaf_node(
                    bs_current_assets.get("cash_and_cash_equivalents", {"value": None, "confidence_score": 0.0})
                ),
                "accounts_receivable": _normalize_leaf_node(
                    bs_current_assets.get("accounts_receivable", {"value": None, "confidence_score": 0.0})
                ),
                "inventory": _normalize_leaf_node(
                    bs_current_assets.get("inventory", {"value": None, "confidence_score": 0.0})
                ),
            },
            "non_current_assets": {
                "property_plant_equipment": _normalize_leaf_node(
                    bs_non_current_assets.get("property_plant_equipment", {"value": None, "confidence_score": 0.0})
                ),
            },
            "current_liabilities": {
                "accounts_payable": _normalize_leaf_node(
                    bs_current_liabilities.get("accounts_payable", {"value": None, "confidence_score": 0.0})
                ),
            },
            "equity": {
                "owner_equity": _normalize_leaf_node(
                    bs_equity.get("owner_equity", {"value": None, "confidence_score": 0.0})
                ),
                "retained_earnings": _normalize_leaf_node(
                    bs_equity.get("retained_earnings", {"value": None, "confidence_score": 0.0})
                ),
            },
            "totals": {
                "total_assets": _normalize_leaf_node(
                    bs_totals.get("total_assets", {"value": None, "confidence_score": 0.0})
                ),
                "total_liabilities": _normalize_leaf_node(
                    bs_totals.get("total_liabilities", {"value": None, "confidence_score": 0.0})
                ),
                "total_equity": _normalize_leaf_node(
                    bs_totals.get("total_equity", {"value": None, "confidence_score": 0.0})
                ),
            },
        },
        "income_statement": {
            "revenue": _normalize_leaf_node(is_raw.get("revenue", {"value": None, "confidence_score": 0.0})),
            "cost_of_goods_sold": _normalize_leaf_node(
                is_raw.get("cost_of_goods_sold", {"value": None, "confidence_score": 0.0})
            ),
            "operating_expenses": _normalize_leaf_node(
                is_raw.get("operating_expenses", {"value": None, "confidence_score": 0.0})
            ),
            "net_income": _normalize_leaf_node(
                is_raw.get("net_income", {"value": None, "confidence_score": 0.0})
            ),
        },
        "cash_flow_statement": {
            "net_cash_from_operations": _normalize_leaf_node(
                cf_raw.get("net_cash_from_operations", {"value": None, "confidence_score": 0.0})
            ),
            "net_cash_from_investing": _normalize_leaf_node(
                cf_raw.get("net_cash_from_investing", {"value": None, "confidence_score": 0.0})
            ),
            "net_cash_from_financing": _normalize_leaf_node(
                cf_raw.get("net_cash_from_financing", {"value": None, "confidence_score": 0.0})
            ),
        },
    }

    return _add_parent_confidence(normalized)


def build_tax_payload(
    raw_result: Dict[str, Any],
    *,
    schema_id: str,
) -> Dict[str, Any]:
    evidence = build_evidence_pack(
        raw_result,
        max_chars=_safe_int(os.getenv("LLM_MAX_EVIDENCE_CHARS", "35000"), 35000),
    )
    messages = _make_tax_prompt(evidence, schema_id=schema_id)
    parsed = _llm_json(messages, max_tokens=3200)
    payload = parsed.get("payload", {})
    return _normalize_tax_payload(payload if isinstance(payload, dict) else {})


def build_financial_payload(
    raw_result: Dict[str, Any],
    *,
    schema_id: str,
) -> Dict[str, Any]:
    evidence = build_evidence_pack(
        raw_result,
        max_chars=_safe_int(os.getenv("LLM_MAX_EVIDENCE_CHARS", "35000"), 35000),
    )
    messages = _make_financial_prompt(evidence, schema_id=schema_id)
    parsed = _llm_json(messages, max_tokens=3500)
    payload = parsed.get("payload", {})
    return _normalize_financial_payload(payload if isinstance(payload, dict) else {})


def build_generic_payload(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    pages = list(ContentUnderstandingClient.iter_pages(raw_result))
    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(raw_result, aggregate_mode="mean")
    table_cells = ContentUnderstandingClient.extract_table_cells_with_confidence(raw_result, aggregate_mode="mean")

    titles: List[Dict[str, Any]] = []
    first_page_lines = [x for x in lines if x.get("page") == 1 and (x.get("text") or "").strip()]
    first_page_lines.sort(key=lambda x: x.get("line_index", 999999))
    for idx, line in enumerate(first_page_lines[:5]):
        titles.append(
            {
                "value": line.get("text"),
                "confidence_score": round(_safe_float(line.get("confidence")) or 0.0, 3),
                "page": line.get("page"),
                "title_index": idx,
            }
        )

    paragraph_items: List[Dict[str, Any]] = []
    source_paragraphs = paragraphs if paragraphs else lines
    for idx, row in enumerate(source_paragraphs):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        paragraph_items.append(
            {
                "value": text,
                "confidence_score": round(_safe_float(row.get("confidence")) or 0.0, 3),
                "page": row.get("page"),
                "paragraph_index": idx,
            }
        )

    tables_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for cell in table_cells:
        key = (_safe_int(cell.get("page"), 0), _safe_int(cell.get("table_index"), 0))
        tables_map.setdefault(key, []).append(cell)

    tables: List[Dict[str, Any]] = []
    for (page, table_index), cells in sorted(tables_map.items(), key=lambda x: (x[0][0], x[0][1])):
        max_row = max((_safe_int(c.get("row_index"), 0) for c in cells), default=-1)
        max_col = max((_safe_int(c.get("column_index"), 0) for c in cells), default=-1)

        rows: List[List[Dict[str, Any]]] = []
        confs: List[float] = []

        for r in range(max_row + 1):
            row_items: List[Dict[str, Any]] = []
            for c in range(max_col + 1):
                matched = next(
                    (
                        x for x in cells
                        if _safe_int(x.get("row_index"), -1) == r and _safe_int(x.get("column_index"), -1) == c
                    ),
                    None,
                )
                if matched is None:
                    row_items.append({"value": None, "confidence_score": 0.0})
                else:
                    conf = round(_safe_float(matched.get("confidence")) or 0.0, 3)
                    row_items.append(
                        {
                            "value": matched.get("text"),
                            "confidence_score": conf,
                        }
                    )
                    if (matched.get("text") or "").strip():
                        confs.append(conf)
            rows.append(row_items)

        tables.append(
            {
                "page": page,
                "table_index": table_index,
                "rows": rows,
                "confidence_score": round(_mean(confs) or 0.0, 3),
            }
        )

    payload = {
        "titles": titles,
        "paragraphs": paragraph_items,
        "tables": tables,
        "page_count": {
            "value": len(pages),
            "confidence_score": 1.0,
        },
    }

    return _add_parent_confidence(payload)


def build_dynamic_document_envelope(
    raw_result: Dict[str, Any],
    *,
    doc_id: str,
    created_utc: str,
    source_blob: str,
    ir_blob: str,
    cu_analyzer_id: str = "prebuilt-layout",
) -> Dict[str, Any]:
    """
    Main entry point for your app.

    It:
    1 detects the schema dynamically
    2 builds the right payload
    3 returns a fixed common envelope
    """
    pages = list(ContentUnderstandingClient.iter_pages(raw_result))
    route = detect_document_schema(raw_result)

    schema_id = route["schema_id"]
    schema_version = route["schema_version"]
    document_type = route["document_type"]
    document_subtype = route["document_subtype"]
    routing_confidence = route["confidence_score"]

    if schema_id.startswith("tax_"):
        payload = build_tax_payload(raw_result, schema_id=schema_id)
    elif schema_id == "financial_statement":
        payload = build_financial_payload(raw_result, schema_id=schema_id)
    else:
        payload = build_generic_payload(raw_result)

    envelope = {
        "metadata": {
            "document_title": _extract_document_title(raw_result),
            "doc_id": doc_id,
            "created_utc": created_utc,
            "source_blob": source_blob,
            "ir_blob": ir_blob,
            "cu_analyzer_id": cu_analyzer_id,
            "document_type": document_type,
            "document_subtype": document_subtype,
            "routing_confidence_score": routing_confidence,
            "chunking": {
                "chunks": len(pages),
            },
        },
        "schema": {
            "schema_id": schema_id,
            "schema_version": schema_version,
        },
        "payload": payload,
    }

    return envelope