"""
Azure Content Understanding client wrapper.

Focus
1 Submit and poll analysis jobs
2 Return raw JSON
3 Provide production ready helpers to extract confidence from prebuilt layout output

Key idea
The service gives confidence reliably at the OCR word level
For lines, paragraphs, and table cells, confidence is derived by aggregating the words that overlap the same span
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


class ContentUnderstandingError(Exception):
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


def _overlaps(a: Span, b: Span) -> bool:
    return not (a.end <= b.offset or b.end <= a.offset)


def _coalesce(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _extract_spans(node: Dict[str, Any]) -> List[Span]:
    spans_raw = node.get("spans")
    if isinstance(spans_raw, list) and spans_raw:
        spans: List[Span] = []
        for s in spans_raw:
            if isinstance(s, dict):
                spans.append(
                    Span(
                        offset=_safe_int(s.get("offset"), 0),
                        length=_safe_int(s.get("length"), 0),
                    )
                )
        return [s for s in spans if s.length > 0]

    span_raw = node.get("span")
    if isinstance(span_raw, dict):
        span = Span(
            offset=_safe_int(span_raw.get("offset"), 0),
            length=_safe_int(span_raw.get("length"), 0),
        )
        return [span] if span.length > 0 else []

    return []


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _min(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return min(values)


class ContentUnderstandingClient:
    """
    Thin wrapper around the Azure Content Understanding REST API.

    Notes
    The API can return either
    A 200 with a complete result JSON
    Or a 202 with an Operation Location header to poll
    """

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

    def list_analyzers(self) -> List[Dict[str, Any]]:
        url = f"{self.endpoint}/contentunderstanding/analyzers?api-version={self.api_version}"
        resp = self._session.get(url, headers=self._headers, timeout=30)
        if resp.status_code != 200:
            raise ContentUnderstandingError(
                f"Failed to list analyzers {resp.status_code} {resp.text}"
            )
        data = resp.json()
        if isinstance(data, dict):
            value = data.get("value")
            return value if isinstance(value, list) else [data]
        if isinstance(data, list):
            return data
        return []

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

        resp = self._session.post(url, headers=headers, data=file_bytes, timeout=120)

        if resp.status_code not in (200, 202):
            raise ContentUnderstandingError(
                f"Submission failed {resp.status_code} {resp.text}"
            )

        op = resp.headers.get("Operation-Location") or resp.headers.get("operation-location")
        if op:
            return op, None

        if resp.status_code == 200:
            return None, resp.json()

        raise ContentUnderstandingError("Missing Operation Location header")

    def _poll(self, operation_url: str) -> Dict[str, Any]:
        if not operation_url:
            raise ContentUnderstandingError("Missing operation url")

        deadline = time.time() + self.max_poll_seconds

        while True:
            if time.time() > deadline:
                raise ContentUnderstandingError("Timed out waiting for analysis result")

            resp = self._session.get(operation_url, headers=self._headers, timeout=60)
            if resp.status_code != 200:
                raise ContentUnderstandingError(
                    f"Polling failed {resp.status_code} {resp.text}"
                )

            data = resp.json()
            status = str(data.get("status", "")).lower()

            if status == "succeeded":
                return data
            if status in ("failed", "canceled"):
                err = data.get("error") or {}
                msg = err.get("message") or "unknown error"
                raise ContentUnderstandingError(f"Analysis {status} {msg}")

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
        """
        Return word rows from layout output

        Each row includes
        page
        text
        confidence
        offset
        length
        source
        """
        rows: List[Dict[str, Any]] = []
        for page in ContentUnderstandingClient.iter_pages(result):
            page_number = _safe_int(_coalesce(page.get("pageNumber"), page.get("page")), 0)
            words = page.get("words") or []
            if not isinstance(words, list):
                continue
            for w in words:
                if not isinstance(w, dict):
                    continue
                spans = _extract_spans(w)
                span = spans[0] if spans else Span(0, 0)
                rows.append(
                    {
                        "page": page_number,
                        "text": w.get("content", ""),
                        "confidence": _safe_float(w.get("confidence")),
                        "offset": span.offset,
                        "length": span.length,
                        "source": w.get("source"),
                    }
                )
        return rows

    @staticmethod
    def _words_index_by_page(result: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for row in ContentUnderstandingClient.extract_words(result):
            p = _safe_int(row.get("page"), 0)
            by_page.setdefault(p, []).append(row)

        for p, rows in by_page.items():
            rows.sort(key=lambda r: _safe_int(r.get("offset"), 0))
            by_page[p] = rows
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

        confs: List[float] = []
        for w in words:
            w_conf = _safe_float(w.get("confidence"))
            if w_conf is None:
                continue
            w_span = Span(_safe_int(w.get("offset"), 0), _safe_int(w.get("length"), 0))
            if w_span.length <= 0:
                continue
            for s in spans:
                if _overlaps(w_span, s):
                    confs.append(w_conf)
                    break

        if not confs:
            return None

        if mode == "min":
            return _min(confs)
        return _mean(confs)

    @staticmethod
    def extract_lines_with_confidence(
        result: Dict[str, Any],
        *,
        aggregate_mode: str = "mean",
    ) -> List[Dict[str, Any]]:
        """
        Return line rows with derived confidence from overlapping words
        """
        rows: List[Dict[str, Any]] = []
        words_by_page = ContentUnderstandingClient._words_index_by_page(result)

        for page in ContentUnderstandingClient.iter_pages(result):
            page_number = _safe_int(_coalesce(page.get("pageNumber"), page.get("page")), 0)
            page_words = words_by_page.get(page_number, [])

            lines = page.get("lines") or []
            if not isinstance(lines, list):
                continue

            for idx, line in enumerate(lines):
                if not isinstance(line, dict):
                    continue
                spans = _extract_spans(line)
                derived = ContentUnderstandingClient._aggregate_confidence_for_spans(
                    words=page_words,
                    spans=spans,
                    mode=aggregate_mode,
                )
                rows.append(
                    {
                        "page": page_number,
                        "line_index": idx,
                        "text": line.get("content", ""),
                        "confidence": derived,
                        "source": line.get("source"),
                        "spans": [{"offset": s.offset, "length": s.length} for s in spans],
                    }
                )
        return rows

    @staticmethod
    def extract_table_cells_with_confidence(
        result: Dict[str, Any],
        *,
        aggregate_mode: str = "mean",
    ) -> List[Dict[str, Any]]:
        """
        Return table cell rows with derived confidence from overlapping words

        Output columns include
        page
        table_index
        row_index
        column_index
        text
        confidence
        """
        rows: List[Dict[str, Any]] = []
        words_by_page = ContentUnderstandingClient._words_index_by_page(result)

        for page in ContentUnderstandingClient.iter_pages(result):
            page_number = _safe_int(_coalesce(page.get("pageNumber"), page.get("page")), 0)
            page_words = words_by_page.get(page_number, [])
            tables = page.get("tables") or []
            if not isinstance(tables, list):
                continue

            for t_idx, table in enumerate(tables):
                if not isinstance(table, dict):
                    continue
                cells = table.get("cells") or []
                if not isinstance(cells, list):
                    continue

                for cell in cells:
                    if not isinstance(cell, dict):
                        continue
                    spans = _extract_spans(cell)
                    derived = ContentUnderstandingClient._aggregate_confidence_for_spans(
                        words=page_words,
                        spans=spans,
                        mode=aggregate_mode,
                    )
                    rows.append(
                        {
                            "page": page_number,
                            "table_index": t_idx,
                            "row_index": _safe_int(cell.get("rowIndex"), 0),
                            "column_index": _safe_int(cell.get("columnIndex"), 0),
                            "text": cell.get("content", ""),
                            "confidence": derived,
                            "kind": cell.get("kind"),
                            "spans": [{"offset": s.offset, "length": s.length} for s in spans],
                        }
                    )
        return rows

    @staticmethod
    def extract_fields_with_confidence(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten documents fields if present

        This is useful for custom analyzers that return documents and fields
        For prebuilt layout, documents and fields are often not present
        """
        rows: List[Dict[str, Any]] = []

        root = ContentUnderstandingClient._root_result(result)
        analyze_result = root.get("analyzeResult") if isinstance(root.get("analyzeResult"), dict) else root
        documents = analyze_result.get("documents") if isinstance(analyze_result, dict) else None
        if not isinstance(documents, list):
            return rows

        for doc_idx, document in enumerate(documents):
            if not isinstance(document, dict):
                continue
            fields = document.get("fields") or {}
            if not isinstance(fields, dict):
                continue
            ContentUnderstandingClient._flatten_fields(fields, "", doc_idx, rows)

        return rows

    @staticmethod
    def _flatten_fields(
        fields: Dict[str, Any],
        parent_key: str,
        doc_idx: int,
        rows: List[Dict[str, Any]],
    ) -> None:
        for field_name, field_data in fields.items():
            full_key = f"{parent_key}.{field_name}" if parent_key else field_name
            if not isinstance(field_data, dict):
                continue

            field_type = str(field_data.get("type") or "")
            confidence = _safe_float(field_data.get("confidence"))
            value_key = f"value{field_type.capitalize()}" if field_type else None

            if field_type == "object":
                nested = field_data.get("valueObject") or {}
                if isinstance(nested, dict):
                    ContentUnderstandingClient._flatten_fields(nested, full_key, doc_idx, rows)
                continue

            if field_type == "array":
                value_array = field_data.get("valueArray") or []
                if not isinstance(value_array, list):
                    continue
                for i, item in enumerate(value_array):
                    if isinstance(item, dict) and str(item.get("type") or "") == "object":
                        nested = item.get("valueObject") or {}
                        if isinstance(nested, dict):
                            ContentUnderstandingClient._flatten_fields(
                                nested, f"{full_key}[{i}]", doc_idx, rows
                            )
                    elif isinstance(item, dict):
                        rows.append(
                            {
                                "document": doc_idx,
                                "field": f"{full_key}[{i}]",
                                "value": item.get("content", ""),
                                "confidence": _safe_float(item.get("confidence")),
                                "type": str(item.get("type") or ""),
                            }
                        )
                    else:
                        rows.append(
                            {
                                "document": doc_idx,
                                "field": f"{full_key}[{i}]",
                                "value": item,
                                "confidence": None,
                                "type": "unknown",
                            }
                        )
                continue

            raw_value = field_data.get(value_key) if value_key else field_data.get("content")
            if raw_value is None:
                raw_value = field_data.get("content")

            rows.append(
                {
                    "document": doc_idx,
                    "field": full_key,
                    "value": raw_value,
                    "confidence": confidence,
                    "type": field_type,
                }
            )