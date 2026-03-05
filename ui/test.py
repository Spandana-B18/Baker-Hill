"""
Azure Content Understanding client wrapper.
Handles document submission, polling, and result parsing.
"""
 
import time
import requests
from typing import Any
 
 
class ContentUnderstandingError(Exception):
    pass
 
 
class ContentUnderstandingClient:
    """
    Thin wrapper around the Azure Content Understanding REST API.
    Docs: https://learn.microsoft.com/azure/ai-services/content-understanding/
    """
 
    POLL_INTERVAL_SEC = 2
    MAX_POLL_RETRIES = 60  # 2 min total
 
    def __init__(self, endpoint: str, api_key: str, api_version: str):
        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.headers = {
            "Ocp-Apim-Subscription-Key": api_key,
        }
 
    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
 
    def list_analyzers(self) -> list[dict[str, Any]]:
        """Return all analyzers available on this Content Understanding resource."""
        url = f"{self.endpoint}/contentunderstanding/analyzers?api-version={self.api_version}"
        response = requests.get(url, headers=self.headers, timeout=30)
        if response.status_code != 200:
            raise ContentUnderstandingError(
                f"Failed to list analyzers [{response.status_code}]: {response.text}"
            )
        data = response.json()
        return data.get("value", data) if isinstance(data, dict) else data
 
    def analyze_document(
        self,
        analyzer_id: str,
        file_bytes: bytes,
        file_name: str,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """
        Submit a document for analysis and poll until complete.
 
        Returns the full result dict from the API.
        """
        operation_url, immediate_result = self._submit(analyzer_id, file_bytes, file_name, content_type)
        if immediate_result is not None:
            return immediate_result
        result = self._poll(operation_url)
        return result
 
    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
 
    def _submit(
        self,
        analyzer_id: str,
        file_bytes: bytes,
        file_name: str,
        content_type: str,
    ) -> str:
        """POST the document and return the operation-location URL."""
        url = (
            f"{self.endpoint}/contentunderstanding/analyzers"
            f"/{analyzer_id}:analyze?api-version={self.api_version}"
        )
 
        headers = {
            **self.headers,
            "Content-Type": content_type,
        }
 
        response = requests.post(url, headers=headers, data=file_bytes, timeout=60)
 
        if response.status_code not in (200, 202):
            raise ContentUnderstandingError(
                f"Submission failed [{response.status_code}]: {response.text}"
            )
 
        operation_url = response.headers.get("Operation-Location") or response.headers.get(
            "operation-location"
        )
        if not operation_url:
            # Some versions return the result immediately (200)
            if response.status_code == 200:
                return None, response.json()
            raise ContentUnderstandingError(
                "No Operation-Location header in response."
            )
 
        return operation_url, None
 
    def _poll(self, operation_url: str) -> dict[str, Any]:
        """Poll until the operation completes and return the result."""
 
        for _ in range(self.MAX_POLL_RETRIES):
            response = requests.get(operation_url, headers=self.headers, timeout=30)
 
            if response.status_code != 200:
                raise ContentUnderstandingError(
                    f"Polling failed [{response.status_code}]: {response.text}"
                )
 
            data = response.json()
            status = data.get("status", "").lower()
 
            if status == "succeeded":
                return data
            if status in ("failed", "canceled"):
                error = data.get("error", {})
                raise ContentUnderstandingError(
                    f"Analysis {status}: {error.get('message', 'unknown error')}"
                )
 
            time.sleep(self.POLL_INTERVAL_SEC)
 
        raise ContentUnderstandingError("Timed out waiting for analysis result.")
 
    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def extract_fields_with_confidence(result: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Walk the result tree and return a flat list of
        { field, value, confidence, page } dicts.
        """
        rows: list[dict[str, Any]] = []
 
        # Navigate: result -> analyzeResult -> documents[]
        analyze_result = result.get("result", result)  # handle both shapes
        documents = (
            analyze_result.get("analyzeResult", {}).get("documents")
            or analyze_result.get("documents")
            or []
        )
 
        for doc_idx, document in enumerate(documents):
            fields: dict = document.get("fields", {})
            ContentUnderstandingClient._flatten_fields(
                fields, parent_key="", doc_idx=doc_idx, rows=rows
            )
 
        return rows
 
    @staticmethod
    def _flatten_fields(
        fields: dict,
        parent_key: str,
        doc_idx: int,
        rows: list[dict[str, Any]],
    ):
        """Recursively flatten nested field structures."""
        for field_name, field_data in fields.items():
            full_key = f"{parent_key}.{field_name}" if parent_key else field_name
 
            if not isinstance(field_data, dict):
                continue
 
            field_type = field_data.get("type", "")
            confidence = field_data.get("confidence")
            value_key = f"value{field_type.capitalize()}" if field_type else None
 
            # Nested object
            if field_type == "object":
                nested = field_data.get("valueObject", {})
                ContentUnderstandingClient._flatten_fields(
                    nested, parent_key=full_key, doc_idx=doc_idx, rows=rows
                )
                continue
 
            # Array of objects
            if field_type == "array":
                for i, item in enumerate(field_data.get("valueArray", [])):
                    if isinstance(item, dict) and item.get("type") == "object":
                        ContentUnderstandingClient._flatten_fields(
                            item.get("valueObject", {}),
                            parent_key=f"{full_key}[{i}]",
                            doc_idx=doc_idx,
                            rows=rows,
                        )
                    else:
                        rows.append(
                            {
                                "document": doc_idx,
                                "field": f"{full_key}[{i}]",
                                "value": item.get("content", ""),
                                "confidence": item.get("confidence"),
                                "type": item.get("type", ""),
                            }
                        )
                continue
 
            # Scalar value
            raw_value = (
                field_data.get(value_key)
                if value_key
                else field_data.get("content")
            )
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
 