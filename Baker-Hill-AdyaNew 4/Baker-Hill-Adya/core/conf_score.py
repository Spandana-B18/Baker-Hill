"""
conf_score.py

Core Content Understanding and confidence scoring logic.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _is_numeric_text(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False
    t = t.replace(",", "").replace("$", "").replace("%", "")
    t = t.replace("(", "-").replace(")", "")
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", t))


def _parse_numeric_text(text: str) -> Optional[float]:
    if not _is_numeric_text(text):
        return None
    t = text.strip().replace(",", "").replace("$", "").replace("%", "")
    t = t.replace("(", "-").replace(")", "")
    try:
        return float(t)
    except Exception:
        return None


def _lower(text: Any) -> str:
    return str(text or "").lower()


def _make_leaf(value: Any, confidence_score: Optional[float]) -> Dict[str, Any]:
    return {
        "value": value,
        "confidence_score": round(float(confidence_score or 0.0), 3),
    }


def _normalize_leaf_node(node: Any) -> Dict[str, Any]:
    if isinstance(node, dict) and "value" in node and "confidence_score" in node:
        conf = _safe_float(node.get("confidence_score"))
        return {
            "value": node.get("value"),
            "confidence_score": round(conf or 0.0, 3),
        }
    return {
        "value": node,
        "confidence_score": 0.0,
    }


def _collect_child_confidences(node: Any) -> List[float]:
    values: List[float] = []
    if isinstance(node, dict):
        if "confidence_score" in node and isinstance(node.get("confidence_score"), (int, float)):
            values.append(float(node["confidence_score"]))
        for key, value in node.items():
            if key == "confidence_score":
                continue
            values.extend(_collect_child_confidences(value))
    elif isinstance(node, list):
        for item in node:
            values.extend(_collect_child_confidences(item))
    return values


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
        if session is not None:
            self._session = session
        else:
            retry_strategy = Retry(
                total=5,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session = requests.Session()
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
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
        max_conn_retries = 5

        while True:
            if time.time() > deadline:
                raise ContentUnderstandingError("Timed out waiting for analysis result")

            conn_attempts = 0
            response = None
            while conn_attempts < max_conn_retries:
                try:
                    response = self._session.get(
                        operation_url, headers=self._headers, timeout=120
                    )
                    break
                except requests.exceptions.ConnectionError as exc:
                    conn_attempts += 1
                    if conn_attempts >= max_conn_retries:
                        raise ContentUnderstandingError(
                            f"Connection lost after {max_conn_retries} retries: {exc}"
                        ) from exc
                    wait = 2 ** conn_attempts
                    time.sleep(wait)

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
        max_lines_per_page: int = 400,
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
        max_paragraphs_per_page: int = 300,
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
        max_tables_per_page: int = 100,
        max_cells_per_table: int = 1200,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        words_by_page = ContentUnderstandingClient._words_index_by_page(result)

        # ── Path A: per-page tables (older Azure CU API format) ──────────────
        # In this format each page object contains a "tables" list.
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

                    # Extract physical X-center from bounding box for layout-aware pairing
                    x_center = None
                    bounding = cell.get("boundingRegions") or []
                    if bounding and isinstance(bounding, list):
                        poly = (bounding[0] if isinstance(bounding[0], dict) else {}).get("polygon") or []
                        if len(poly) >= 4:
                            # polygon = [x1,y1, x2,y2, x3,y3, x4,y4] — corners TL,TR,BR,BL
                            xs = [poly[i] for i in range(0, len(poly), 2)]
                            x_center = sum(xs) / len(xs)

                    rows.append(
                        {
                            "page": page_number,
                            "table_index": table_idx,
                            "cell_index": cell_idx,
                            "row_index": _safe_int(cell.get("rowIndex"), 0),
                            "column_index": _safe_int(cell.get("columnIndex"), 0),
                            "column_span": _safe_int(cell.get("columnSpan"), 1),
                            "kind": cell.get("kind"),
                            "text": cell.get("content", ""),
                            "confidence": confidence,
                            "x_center": x_center,
                            "spans": [{"offset": s.offset, "length": s.length} for s in spans],
                        }
                    )

        # ── Path B: content-level tables (newer Azure CU API format) ─────────
        # In this format tables live at contents[i]["tables"], NOT inside page
        # objects. Each cell carries a "source" field encoding the page number
        # and bounding polygon: D(page, x1,y1, x2,y2, x3,y3, x4,y4).
        if not rows:
            _src_re = re.compile(
                r"D\((\d+),([\d.]+),([\d.]+),([\d.]+),([\d.]+)"
                r",([\d.]+),([\d.]+),([\d.]+),([\d.]+)\)"
            )
            root = ContentUnderstandingClient._root_result(result)
            for content in (root.get("contents") or []):
                if not isinstance(content, dict):
                    continue
                tables = content.get("tables") or []
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

                        # Parse page number and x_center from source string
                        src = cell.get("source") or ""
                        m = _src_re.match(src)
                        if m:
                            page_number = int(m.group(1))
                            xs = [float(m.group(i)) for i in (2, 4, 6, 8)]
                            x_center: Optional[float] = sum(xs) / len(xs)
                        else:
                            page_number = 0
                            x_center = None

                        page_words = words_by_page.get(page_number, [])
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
                                "column_span": _safe_int(cell.get("columnSpan"), 1),
                                "kind": cell.get("kind"),
                                "text": cell.get("content", ""),
                                "confidence": confidence,
                                "x_center": x_center,
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
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

    if not endpoint or not api_key or not deployment or not api_version:
        raise LLMError("Missing Azure OpenAI configuration in environment variables")

    url = (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={api_version}"
    )

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
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


def _llm_json(messages: List[Dict[str, Any]], *, max_tokens: int = 3500) -> Dict[str, Any]:
    text = _azure_openai_chat_completion(messages=messages, temperature=0.0, max_tokens=max_tokens)
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise LLMError(f"LLM did not return valid JSON: {exc}")

    if not isinstance(parsed, dict):
        raise LLMError("LLM root must be a JSON object")

    return parsed


def _compact_lines(lines: List[Dict[str, Any]], limit: int = 500) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in lines[:limit]:
        text = _normalize_space(row.get("text", ""))
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
        text = _normalize_space(row.get("text", ""))
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


def _compact_table_cells(cells: List[Dict[str, Any]], limit: int = 1500) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in cells[:limit]:
        text = _normalize_space(row.get("text", ""))
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

    # Page 1 header lines in reading order — used to anchor entity-level fields
    # so the LLM sees the form header before any Schedule B investment tables.
    page_1_lines_ordered = sorted(
        [l for l in lines if _safe_int(l.get("page"), 0) == 1],
        key=lambda l: (_safe_int(l.get("line_index"), 999999)),
    )
    page_1_header = _compact_lines(page_1_lines_ordered, limit=60)

    pack = {
        "page_1_header": page_1_header,
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
        "page_1_header": pack.get("page_1_header", []),
        "lines": pack["lines"][:keep_lines],
        "paragraphs": pack["paragraphs"][:keep_paragraphs],
        "table_cells": pack["table_cells"][:keep_cells],
    }


def detect_document_schema(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect document type and subtype from the extracted CU result.

    Detection order:
    1. Exact tax form regex on first page and early document text
    2. Weighted keyword scoring as fallback
    3. Financial statement detection
    4. Generic fallback
    """
    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(
        raw_result,
        aggregate_mode="mean",
    )

    first_page_text_parts: List[str] = []
    early_text_parts: List[str] = []

    for row in lines[:180]:
        text = _normalize_space(row.get("text", ""))
        if not text:
            continue
        early_text_parts.append(text)
        if _safe_int(row.get("page"), 0) == 1:
            first_page_text_parts.append(text)

    for row in paragraphs[:80]:
        text = _normalize_space(row.get("text", ""))
        if not text:
            continue
        early_text_parts.append(text)
        if _safe_int(row.get("page"), 0) == 1:
            first_page_text_parts.append(text)

    first_page_text = "\n".join(first_page_text_parts).lower()
    early_text = "\n".join(early_text_parts).lower()

    def _result(
        *,
        document_type: str,
        document_subtype: str,
        schema_id: str,
        confidence_score: float,
    ) -> Dict[str, Any]:
        return {
            "document_type": document_type,
            "document_subtype": document_subtype,
            "schema_id": schema_id,
            "schema_version": "2024",
            "confidence_score": round(confidence_score, 3),
        }

    exact_tax_rules: List[Tuple[str, str, str, List[str], float]] = [
        # Main form rules MUST come before k1 so that a 1065/1120-S document
        # that contains an embedded Schedule K-1 is classified by its primary
        # form, not by the embedded schedule.
        (
            "1120s",
            "tax_1120s",
            r"\bform\s*1120[\-\s]*s\b|\b1120[\-\s]*s\b|u\.s\.\s+income\s+tax\s+return\s+for\s+an\s+s\s+corporation",
            ["schedule l", "s corporation", "ordinary business income"],
            0.97,
        ),
        (
            "1065",
            "tax_1065",
            r"\bform\s*1065\b|u\.s\.\s+return\s+of\s+partnership\s+income",
            ["partnership", "ordinary business income", "schedule b"],
            0.97,
        ),
        (
            "1040",
            "tax_1040",
            r"\bform\s*1040\b|u\.s\.\s+individual\s+income\s+tax\s+return",
            ["filing status", "adjusted gross income", "dependents"],
            0.97,
        ),
        (
            "1120",
            "tax_1120",
            r"\bform\s*1120\b|u\.s\.\s+corporation\s+income\s+tax\s+return",
            ["corporation income tax return", "schedule l", "taxable income"],
            0.97,
        ),
        # k1 is checked LAST — only matches standalone Schedule K-1 documents
        # where no parent form (1065, 1120-S) was found first.
        (
            "k1",
            "tax_k1",
            r"\bschedule\s*k[\-\s]*1\b",
            ["shareholder's share", "partner's share", "beneficiary's share"],
            0.97,
        ),
    ]

    # ------------------------------------------------------------------
    # Schedule detection: finds all schedules present in the document
    # so the subtype can reflect e.g. "Form 1065 with Schedule K-1"
    # ------------------------------------------------------------------
    _SCHEDULE_RULES: List[Tuple[str, str]] = [
        # (regex pattern, human-readable label)
        (r"\bschedule\s*k[\-\s]*1\b",           "Schedule K-1"),
        (r"\bschedule\s*[lL]\b",                 "Schedule L"),
        (r"\bschedule\s*[mM][\-\s]*1\b",         "Schedule M-1"),
        (r"\bschedule\s*[mM][\-\s]*2\b",         "Schedule M-2"),
        (r"\bschedule\s*[mM][\-\s]*3\b",         "Schedule M-3"),
        (r"\bschedule\s*[bB]\b",                 "Schedule B"),
        (r"\bschedule\s*[dD]\b",                 "Schedule D"),
        (r"\bschedule\s*[eE]\b",                 "Schedule E"),
        (r"\bschedule\s*[cC]\b",                 "Schedule C"),
        (r"\bschedule\s*[aA]\b",                 "Schedule A"),
        (r"\bschedule\s*[sS][eE]\b",             "Schedule SE"),
    ]

    _FORM_LABEL: Dict[str, str] = {
        "1120s": "Form 1120-S",
        "1065":  "Form 1065",
        "1040":  "Form 1040",
        "1120":  "Form 1120",
        "k1":    "Schedule K-1",
    }

    def _build_subtype(base_subtype: str, schema_id: str) -> str:
        """
        Build a human-readable document_subtype that names the primary form
        and lists any schedules found in the document text.

        Examples:
          "Form 1065 with Schedule K-1, Schedule L"
          "Form 1040 with Schedule C, Schedule E"
          "Form 1120-S"
        """
        base_label = _FORM_LABEL.get(base_subtype, f"Form {base_subtype.upper()}")

        # For standalone K-1 there is nothing else to append
        if base_subtype == "k1":
            return base_label

        schedules_found = []
        for sched_pattern, sched_label in _SCHEDULE_RULES:
            if re.search(sched_pattern, early_text, flags=re.IGNORECASE):
                # Don't list "Schedule K-1" as an extra schedule for 1065/1120-S
                # because K-1 is a normal part of those forms — only call it out
                # when it's genuinely a separately attached K-1 page after the
                # main return, i.e. when the label adds real disambiguation value.
                # We keep it for 1040 (uncommon) and 1120 (uncommon).
                if sched_label == "Schedule K-1" and base_subtype in ("1065", "1120s"):
                    continue
                schedules_found.append(sched_label)

        if schedules_found:
            return f"{base_label} with {', '.join(schedules_found)}"
        return base_label

    for subtype, schema_id, pattern, supporting_terms, confidence in exact_tax_rules:
        if re.search(pattern, first_page_text, flags=re.IGNORECASE):
            support_hits = sum(1 for term in supporting_terms if term in early_text)
            boosted_conf = min(0.99, confidence + (0.005 * support_hits))
            return _result(
                document_type="tax_document",
                document_subtype=_build_subtype(subtype, schema_id),
                schema_id=schema_id,
                confidence_score=boosted_conf,
            )

    for subtype, schema_id, pattern, supporting_terms, confidence in exact_tax_rules:
        if re.search(pattern, early_text, flags=re.IGNORECASE):
            support_hits = sum(1 for term in supporting_terms if term in early_text)
            boosted_conf = min(0.98, 0.93 + (0.01 * support_hits))
            return _result(
                document_type="tax_document",
                document_subtype=_build_subtype(subtype, schema_id),
                schema_id=schema_id,
                confidence_score=boosted_conf,
            )

    tax_patterns: Dict[str, List[Tuple[str, float]]] = {
        "tax_1120s": [
            ("form 1120-s", 5.0),
            ("1120-s", 5.0),
            ("u.s. income tax return for an s corporation", 5.0),
            ("s corporation", 2.5),
            ("schedule l", 1.0),
            ("ordinary business income", 2.0),
            ("shareholders", 1.5),
        ],
        "tax_k1": [
            ("schedule k-1", 5.0),
            ("shareholder's share", 3.0),
            ("partner's share", 3.0),
            ("beneficiary's share", 3.0),
            # NOTE: do NOT add "form 1065 schedule k-1" here — that boosts k1
            # even when the parent Form 1065 is present. The 1065 rule already
            # scores high enough from "form 1065" / "partnership income" hits.
        ],
        "tax_1040": [
            ("form 1040", 5.0),
            ("u.s. individual income tax return", 5.0),
            ("filing status", 2.0),
            ("adjusted gross income", 2.5),
            ("dependents", 1.5),
            ("standard deduction", 1.5),
        ],
        "tax_1065": [
            ("form 1065", 5.0),
            ("u.s. return of partnership income", 5.0),
            ("partnership income", 2.5),
            ("partners", 1.5),
            ("schedule b", 1.0),
        ],
        "tax_1120": [
            ("form 1120", 5.0),
            ("u.s. corporation income tax return", 5.0),
            ("corporation income tax return", 3.0),
            ("taxable income", 1.5),
            ("schedule l", 1.0),
        ],
    }

    financial_patterns: List[Tuple[str, float]] = [
        ("balance sheet", 2.0),
        ("statement of financial position", 3.0),
        ("income statement", 2.5),
        ("statement of cash flows", 2.5),
        ("statement of owner equity", 2.0),
        ("statement of owners equity", 2.0),
        ("ratio analysis", 1.5),
        ("current assets", 1.0),
        ("current liabilities", 1.0),
        ("retained earnings", 1.0),
        ("net income", 1.0),
        ("total assets", 1.0),
        ("total liabilities", 1.0),
        ("cash and cash equivalents", 1.0),
        ("accounts receivable", 1.0),
        ("inventory", 1.0),
    ]

    scores: Dict[str, float] = {
        "tax_1120s": 0.0,
        "tax_k1": 0.0,
        "tax_1040": 0.0,
        "tax_1065": 0.0,
        "tax_1120": 0.0,
        "financial_statement": 0.0,
        "generic_document": 0.0,
    }

    def _add_weighted_hits(target_scores: Dict[str, float], schema_id: str, patterns: List[Tuple[str, float]]) -> None:
        for phrase, weight in patterns:
            if phrase in early_text:
                target_scores[schema_id] += weight
            if phrase in first_page_text:
                target_scores[schema_id] += weight * 0.5

    for schema_id, patterns in tax_patterns.items():
        _add_weighted_hits(scores, schema_id, patterns)

    for phrase, weight in financial_patterns:
        if phrase in early_text:
            scores["financial_statement"] += weight
        if phrase in first_page_text:
            scores["financial_statement"] += weight * 0.5

    best_schema = max(scores, key=scores.get)
    best_score = scores[best_schema]

    if best_score <= 0:
        return _result(
            document_type="generic_document",
            document_subtype="generic",
            schema_id="generic_document",
            confidence_score=0.2,
        )

    if best_schema.startswith("tax_"):
        subtype = best_schema.replace("tax_", "")
        confidence = min(0.94, 0.55 + (0.04 * best_score))
        return _result(
            document_type="tax_document",
            document_subtype=_build_subtype(subtype, best_schema),
            schema_id=best_schema,
            confidence_score=confidence,
        )

    if best_schema == "financial_statement":
        confidence = min(0.94, 0.55 + (0.03 * best_score))
        return _result(
            document_type="financial_document",
            document_subtype="statement",
            schema_id="financial_statement",
            confidence_score=confidence,
        )

    return _result(
        document_type="generic_document",
        document_subtype="generic",
        schema_id="generic_document",
        confidence_score=0.2,
    )


def _extract_document_title(raw_result: Dict[str, Any]) -> Optional[str]:
    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(raw_result, aggregate_mode="mean")
    if paragraphs:
        first_page = [p for p in paragraphs if p.get("page") == 1 and _normalize_space(p.get("text", ""))]
        if first_page:
            first_page.sort(key=lambda x: (x.get("paragraph_index", 999999), -(x.get("confidence") or 0.0)))
            return _normalize_space(first_page[0].get("text", "")) or None

    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    first_page_lines = [l for l in lines if l.get("page") == 1 and _normalize_space(l.get("text", ""))]
    if first_page_lines:
        first_page_lines.sort(key=lambda x: x.get("line_index", 999999))
        return _normalize_space(first_page_lines[0].get("text", "")) or None

    return None


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
Use only the evidence provided in page_1_header and the rest of the document.
If a value cannot be found, set its value to null and confidence_score to 0.

CRITICAL EXTRACTION RULES — read these before extracting any value:

1. entity_name: Extract the PARTNERSHIP/CORPORATION/INDIVIDUAL NAME from the form header on page 1.
   - For Form 1065: look for "Name of partnership" label on page 1 header.
   - For Form 1120-S / 1120: look for "Name" label on page 1 header.
   - For Form 1040: look for taxpayer name on page 1 header.
   - IMPORTANT: Do NOT use any company names from Schedule B investment tables
     (e.g., "Pixar Studios", "Marvel Studios", "Bank of America"). Those are
     subsidiaries or investments — NOT the filing entity.
   - The correct name appears near the top of page 1, next to or below
     "Name of partnership" / "Name" / "Name of corporation".

2. tax_year: Extract the 4-digit calendar year from "For calendar year YYYY"
   near the top of page 1. Also check the large year printed in the top-right
   corner (e.g., "2024").

3. employer_identification_number: The number after "D Employer identification
   number" or "Employer identification number (EIN)" on page 1 header.
   Format: XX-XXXXXXX or 9 digits. NOT the business code number (field C).

4. business_activity_code: Field C "Business code number" on page 1 header.
   This is a numeric code (e.g., "987456311"). NOT field A or field B.

5. principal_business_activity: Field A "Principal business activity" on page 1.
   This is a short description like "Cash", "Retail", "Manufacturing".

6. principal_product_or_service: Field B "Principal product or service" on page 1.

7. date_business_started: Field E "Date business started" on page 1 header.
   Format: MM/DD/YYYY.

8. address: The street/city/state/zip of the filing entity from page 1 header.
   Street is "Number, street, room or suite no." City/state/zip is the next line.

9. key_amounts:
   - total_assets_end_of_year: Field F "Total assets (see instructions)" from
     page 1 header (the dollar amount next to the "$" sign), OR Schedule L
     line 14 column (d) "End of tax year". Use the page 1 field F value first.
   - ordinary_business_income_loss: Line 23 "Ordinary business income (loss)"
     on page 1. This is the bottom-line operating income/loss number.
   - cash: Schedule L line 1 "Cash" — use column (b) Beginning of tax year
     and/or column (d) End of tax year.
   - accounts_receivable: Schedule L line 2a "Trade notes and accounts
     receivable" — column (b) or (d).

Return this structure:

{
  "payload": {
    "form_type": {"value": <string or null>, "confidence_score": <number>},
    "tax_year": {"value": <string or null>, "confidence_score": <number>},
    "entity_name": {"value": <string or null>, "confidence_score": <number>},
    "employer_identification_number": {"value": <string or null>, "confidence_score": <number>},
    "business_activity_code": {"value": <string or null>, "confidence_score": <number>},
    "principal_business_activity": {"value": <string or null>, "confidence_score": <number>},
    "principal_product_or_service": {"value": <string or null>, "confidence_score": <number>},
    "date_business_started": {"value": <string or null>, "confidence_score": <number>},
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
        "principal_business_activity": leaf_at(payload, "principal_business_activity"),
        "principal_product_or_service": leaf_at(payload, "principal_product_or_service"),
        "date_business_started": leaf_at(payload, "date_business_started"),
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


def build_tax_payload(
    raw_result: Dict[str, Any],
    *,
    schema_id: str,
) -> Dict[str, Any]:
    evidence = build_evidence_pack(
        raw_result,
        max_chars=_safe_int(os.getenv("LLM_MAX_EVIDENCE_CHARS", "35000"), 35000),
    )
    parsed = _llm_json(_make_tax_prompt(evidence, schema_id=schema_id), max_tokens=3200)
    payload = parsed.get("payload", {})
    return _normalize_tax_payload(payload if isinstance(payload, dict) else {})


def _group_cells_by_table(cells: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for cell in cells:
        key = (_safe_int(cell.get("page"), 0), _safe_int(cell.get("table_index"), 0))
        grouped.setdefault(key, []).append(cell)
    return grouped


def _table_to_grid(cells: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not cells:
        return []

    max_row = max((_safe_int(c.get("row_index"), 0) for c in cells), default=-1)
    max_col = max((_safe_int(c.get("column_index"), 0) for c in cells), default=-1)

    lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for cell in cells:
        key = (_safe_int(cell.get("row_index"), 0), _safe_int(cell.get("column_index"), 0))
        lookup[key] = cell

    grid: List[List[Dict[str, Any]]] = []
    for r in range(max_row + 1):
        row_items: List[Dict[str, Any]] = []
        for c in range(max_col + 1):
            cell = lookup.get((r, c))
            if cell is None:
                row_items.append(
                    {
                        "text": "",
                        "confidence": 0.0,
                        "row_index": r,
                        "column_index": c,
                    }
                )
            else:
                row_items.append(
                    {
                        "text": _normalize_space(cell.get("text", "")),
                        "confidence": round(_safe_float(cell.get("confidence")) or 0.0, 3),
                        "row_index": r,
                        "column_index": c,
                    }
                )
        grid.append(row_items)
    return grid


def _candidate_statement_text(page: int, table_index: int, lines: List[Dict[str, Any]]) -> str:
    page_lines = [x for x in lines if _safe_int(x.get("page"), 0) == page]
    page_lines.sort(key=lambda x: x.get("line_index", 999999))
    snippets: List[str] = []
    for line in page_lines[:40]:
        text = _normalize_space(line.get("text", ""))
        if text:
            snippets.append(text.lower())
    return "\n".join(snippets)


def _statement_type_from_text(text: str) -> str:
    t = text.lower()
    if "cash flow" in t:
        return "cash_flow_statement"
    if "income statement" in t or "profit and loss" in t or "statement of operations" in t:
        return "income_statement"
    if "owner equity" in t or "owners equity" in t or "statement of equity" in t:
        return "statement_of_owner_equity"
    if "balance sheet" in t or "statement of financial position" in t:
        return "balance_sheet"
    if "ratio analysis" in t:
        return "ratio_analysis"
    return "unknown_statement"


def _guess_statement_title(page: int, table_index: int, grid: List[List[Dict[str, Any]]], lines: List[Dict[str, Any]]) -> str:
    context = _candidate_statement_text(page, table_index, lines)
    statement_type = _statement_type_from_text(context)
    if statement_type != "unknown_statement":
        return statement_type.replace("_", " ")

    for row in grid[:3]:
        joined = " ".join(_normalize_space(c.get("text", "")) for c in row if _normalize_space(c.get("text", "")))
        guess = _statement_type_from_text(joined)
        if guess != "unknown_statement":
            return guess.replace("_", " ")

    return "table"


def _header_candidates(grid: List[List[Dict[str, Any]]]) -> Dict[int, str]:
    headers: Dict[int, str] = {}
    if not grid:
        return headers

    for col_idx in range(len(grid[0])):
        values: List[str] = []
        for r in range(min(3, len(grid))):
            text = _normalize_space(grid[r][col_idx].get("text", ""))
            if text:
                values.append(text)
        candidate = " ".join(values).strip()
        if candidate:
            headers[col_idx] = candidate
    return headers


def _extract_financial_line_items_from_grid(
    grid: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if not grid:
        return []

    headers = _header_candidates(grid)
    items: List[Dict[str, Any]] = []

    for row in grid:
        non_empty = [cell for cell in row if _normalize_space(cell.get("text", ""))]
        if len(non_empty) < 2:
            continue

        label_cell = None
        for cell in row:
            text = _normalize_space(cell.get("text", ""))
            if text and not _is_numeric_text(text):
                label_cell = cell
                break

        if label_cell is None:
            continue

        label_text = _normalize_space(label_cell.get("text", ""))
        if not label_text:
            continue

        values: List[Dict[str, Any]] = []
        row_confidences: List[float] = [round(_safe_float(label_cell.get("confidence")) or 0.0, 3)]

        for cell in row:
            text = _normalize_space(cell.get("text", ""))
            if not text:
                continue
            if cell["column_index"] == label_cell["column_index"]:
                continue
            if _is_numeric_text(text):
                conf = round(_safe_float(cell.get("confidence")) or 0.0, 3)
                row_confidences.append(conf)
                values.append(
                    {
                        "column_name": _make_leaf(headers.get(cell["column_index"], f"column_{cell['column_index']}"), 1.0),
                        "amount": _make_leaf(_parse_numeric_text(text) if _parse_numeric_text(text) is not None else text, conf),
                    }
                )

        if not values:
            continue

        item = {
            "label": _make_leaf(label_text, round(_safe_float(label_cell.get("confidence")) or 0.0, 3)),
            "values": values,
            "confidence_score": round(_mean(row_confidences) or 0.0, 3),
        }
        items.append(item)

    return items


def _find_best_amount(statement_items: List[Dict[str, Any]], patterns: List[str]) -> Dict[str, Any]:
    best_value: Any = None
    best_conf = 0.0

    for statement in statement_items:
        for item in statement.get("line_items", []):
            label = _lower(item.get("label", {}).get("value"))
            if any(p in label for p in patterns):
                for amount_entry in item.get("values", []):
                    amount_leaf = amount_entry.get("amount", {})
                    conf = round(_safe_float(amount_leaf.get("confidence_score")) or 0.0, 3)
                    if conf >= best_conf:
                        best_conf = conf
                        best_value = amount_leaf.get("value")

    return _make_leaf(best_value, best_conf)


def build_financial_payload(raw_result: Dict[str, Any], *, schema_id: str) -> Dict[str, Any]:
    """
    Stronger financial builder.

    It preserves what the document actually contains:
    statements included
    raw statements with line items
    normalized summary when a mapping is possible
    """
    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    cells = ContentUnderstandingClient.extract_table_cells_with_confidence(raw_result, aggregate_mode="mean")
    grouped = _group_cells_by_table(cells)

    statements: List[Dict[str, Any]] = []
    statement_type_leaves: List[Dict[str, Any]] = []

    for (page, table_index), table_cells in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        grid = _table_to_grid(table_cells)
        if not grid:
            continue

        title = _guess_statement_title(page, table_index, grid, lines)
        statement_type = _statement_type_from_text(title)
        line_items = _extract_financial_line_items_from_grid(grid)

        if not line_items and statement_type == "unknown_statement":
            continue

        statement_conf_values: List[float] = []
        for item in line_items:
            statement_conf_values.append(round(_safe_float(item.get("confidence_score")) or 0.0, 3))

        statement_conf = round(_mean(statement_conf_values) or 0.0, 3)

        statements.append(
            {
                "statement_type": _make_leaf(statement_type if statement_type != "unknown_statement" else title, 0.95 if statement_type != "unknown_statement" else 0.6),
                "statement_title": _make_leaf(title, 0.9),
                "page": _make_leaf(page, 1.0),
                "table_index": _make_leaf(table_index, 1.0),
                "line_items": line_items,
                "confidence_score": statement_conf,
            }
        )

        st_leaf = _make_leaf(title, 0.95 if statement_type != "unknown_statement" else 0.6)
        statement_type_leaves.append(st_leaf)

    normalized_summary = {
        "balance_sheet": {
            "current_assets": {
                "cash_and_cash_equivalents": _find_best_amount(
                    statements,
                    ["cash and cash equivalents", "cash checking savings", "cash", "checking savings"],
                ),
                "accounts_receivable": _find_best_amount(
                    statements,
                    ["accounts receivable", "trade accounts receivable", "receivables", "a/r"],
                ),
                "inventory": _find_best_amount(
                    statements,
                    ["inventory", "raw materials", "work in progress", "finished goods"],
                ),
            },
            "non_current_assets": {
                "property_plant_equipment": _find_best_amount(
                    statements,
                    ["property plant equipment", "machinery", "equipment", "buildings", "land"],
                ),
            },
            "current_liabilities": {
                "accounts_payable": _find_best_amount(
                    statements,
                    ["accounts payable", "trade accounts payable", "payables"],
                ),
            },
            "equity": {
                "owner_equity": _find_best_amount(
                    statements,
                    ["owner equity", "owners equity", "capital", "member equity"],
                ),
                "retained_earnings": _find_best_amount(
                    statements,
                    ["retained earnings"],
                ),
            },
            "totals": {
                "total_assets": _find_best_amount(
                    statements,
                    ["total assets"],
                ),
                "total_liabilities": _find_best_amount(
                    statements,
                    ["total liabilities"],
                ),
                "total_equity": _find_best_amount(
                    statements,
                    ["total equity", "owner equity", "owners equity"],
                ),
            },
        },
        "income_statement": {
            "revenue": _find_best_amount(
                statements,
                ["revenue", "sales", "gross receipts", "income"],
            ),
            "cost_of_goods_sold": _find_best_amount(
                statements,
                ["cost of goods sold", "cogs"],
            ),
            "operating_expenses": _find_best_amount(
                statements,
                ["operating expenses", "expenses"],
            ),
            "net_income": _find_best_amount(
                statements,
                ["net income", "net profit", "profit"],
            ),
        },
        "cash_flow_statement": {
            "net_cash_from_operations": _find_best_amount(
                statements,
                ["net cash from operations", "cash provided by operating activities"],
            ),
            "net_cash_from_investing": _find_best_amount(
                statements,
                ["net cash from investing", "cash provided by investing activities"],
            ),
            "net_cash_from_financing": _find_best_amount(
                statements,
                ["net cash from financing", "cash provided by financing activities"],
            ),
        },
    }

    payload = {
        "statements_included": statement_type_leaves,
        "statements": statements,
        "normalized_summary": normalized_summary,
    }

    return _add_parent_confidence(payload)


def build_generic_payload(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    pages = list(ContentUnderstandingClient.iter_pages(raw_result))
    lines = ContentUnderstandingClient.extract_lines_with_confidence(raw_result, aggregate_mode="mean")
    paragraphs = ContentUnderstandingClient.extract_paragraphs_with_confidence(raw_result, aggregate_mode="mean")
    table_cells = ContentUnderstandingClient.extract_table_cells_with_confidence(raw_result, aggregate_mode="mean")

    titles: List[Dict[str, Any]] = []
    first_page_lines = [x for x in lines if x.get("page") == 1 and _normalize_space(x.get("text", ""))]
    first_page_lines.sort(key=lambda x: x.get("line_index", 999999))
    for idx, line in enumerate(first_page_lines[:5]):
        titles.append(
            {
                "value": _normalize_space(line.get("text", "")),
                "confidence_score": round(_safe_float(line.get("confidence")) or 0.0, 3),
                "page": line.get("page"),
                "title_index": idx,
            }
        )

    paragraph_items: List[Dict[str, Any]] = []
    source_paragraphs = paragraphs if paragraphs else lines
    for idx, row in enumerate(source_paragraphs):
        text = _normalize_space(row.get("text", ""))
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
        grid = _table_to_grid(cells)
        confs: List[float] = []

        rows: List[List[Dict[str, Any]]] = []
        for row in grid:
            row_items: List[Dict[str, Any]] = []
            for cell in row:
                conf = round(_safe_float(cell.get("confidence")) or 0.0, 3)
                value = cell.get("text") or None
                if value:
                    confs.append(conf)
                row_items.append(
                    {
                        "value": value,
                        "confidence_score": conf,
                    }
                )
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
        "page_count": _make_leaf(len(pages), 1.0),
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