import json
import os
import time
from http.client import RemoteDisconnected
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .embeddings import AzureOpenAIEmbedder, EmbeddingError


class SearchIndexerError(Exception):
    pass


class AzureAISearchIndexer:
    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        admin_key: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        self.endpoint = (endpoint or os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", "")).rstrip("/")
        self.admin_key = admin_key or os.getenv("AZURE_SEARCH_ADMIN_KEY", "")
        self.api_version = api_version or os.getenv("AZURE_SEARCH_API_VERSION", "2025-09-01")

        self.tax_index = os.getenv("AZURE_SEARCH_TAX_INDEX", "tax-documents-index")
        self.financial_index = os.getenv("AZURE_SEARCH_FINANCIAL_INDEX", "financial-documents-index")
        self.generic_index = os.getenv("AZURE_SEARCH_GENERIC_INDEX", "generic-documents-index")

        self.vector_field = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "content_vector")
        self.content_field = os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content")
        self.id_field = os.getenv("AZURE_SEARCH_ID_FIELD", "id")
        self.vector_dimensions = int(os.getenv("AZURE_SEARCH_VECTOR_DIMENSIONS", "3072"))

        # Safer defaults for vector payload uploads
        self.batch_size = int(os.getenv("AZURE_SEARCH_BATCH_SIZE", "25"))
        self.max_batch_bytes = int(os.getenv("AZURE_SEARCH_MAX_BATCH_BYTES", str(3 * 1024 * 1024)))
        self.max_retries = int(os.getenv("AZURE_SEARCH_MAX_RETRIES", "5"))
        self.timeout_sec = int(os.getenv("AZURE_SEARCH_TIMEOUT_SEC", "180"))
        self.sleep_between_batches_sec = float(os.getenv("AZURE_SEARCH_SLEEP_BETWEEN_BATCHES_SEC", "0.4"))

        if not self.endpoint:
            raise SearchIndexerError("Missing AZURE_SEARCH_SERVICE_ENDPOINT")
        if not self.admin_key:
            raise SearchIndexerError("Missing AZURE_SEARCH_ADMIN_KEY")

        retry_strategy = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.5,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)

        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "api-key": self.admin_key,
                # Helps avoid stale keep alive issues on some services
                "Connection": "close",
            }
        )

    def choose_index_name(self, document_type: str) -> str:
        doc_type = str(document_type or "").strip().lower()

        if doc_type == "tax_document":
            return self.tax_index
        if doc_type == "financial_document":
            return self.financial_index
        return self.generic_index

    def build_index_schema(
        self,
        *,
        index_name: str,
        vector_dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        dims = vector_dimensions or self.vector_dimensions

        return {
            "name": index_name,
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "document_id",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "chunk_id",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "title",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": False,
                    "sortable": True,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "content_vector",
                    "type": "Collection(Edm.Single)",
                    "searchable": True,
                    "retrievable": False,
                    "stored": False,
                    "dimensions": dims,
                    "vectorSearchProfile": "default-vector-profile",
                },
                {
                    "name": "document_type",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "document_subtype",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "schema_id",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "schema_version",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "source_file_name",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": True,
                    "sortable": True,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "source_blob",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "raw_json_blob",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True,
                },
                {
                    "name": "page_number",
                    "type": "Edm.Int32",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "chunk_type",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "line_count",
                    "type": "Edm.Int32",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "confidence_score",
                    "type": "Edm.Double",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": True,
                    "retrievable": True,
                },
                {
                    "name": "processed_at",
                    "type": "Edm.String",
                    "searchable": False,
                    "filterable": True,
                    "sortable": True,
                    "facetable": False,
                    "retrievable": True,
                },
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "default-hnsw",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine",
                        },
                    }
                ],
                "profiles": [
                    {
                        "name": "default-vector-profile",
                        "algorithm": "default-hnsw",
                    }
                ],
            },
            "semantic": {
                "configurations": [
                    {
                        "name": "default-semantic-config",
                        "prioritizedFields": {
                            "titleField": {"fieldName": "title"},
                            "prioritizedContentFields": [
                                {"fieldName": "content"}
                            ],
                        },
                    }
                ]
            },
        }

    def index_exists(self, index_name: str) -> bool:
        url = f"{self.endpoint}/indexes/{index_name}?api-version={self.api_version}"
        response = self.session.get(url, timeout=self.timeout_sec)

        if response.status_code == 200:
            return True
        if response.status_code == 404:
            return False

        raise SearchIndexerError(
            f"Failed checking index existence {response.status_code}: {response.text}"
        )

    def create_index_if_missing(
        self,
        index_name: str,
        *,
        vector_dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.index_exists(index_name):
            return {
                "status": "already_exists",
                "index_name": index_name,
            }

        schema = self.build_index_schema(
            index_name=index_name,
            vector_dimensions=vector_dimensions,
        )

        url = f"{self.endpoint}/indexes?api-version={self.api_version}"
        return self._request_with_retry("POST", url, json_body=schema)

    def index_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        *,
        action: str = "mergeOrUpload",
    ) -> Dict[str, Any]:
        if not documents:
            return {
                "status": "no_documents",
                "uploaded": 0,
                "failed": 0,
                "failed_items": [],
            }

        total_uploaded = 0
        failed_items: List[Dict[str, Any]] = []

        prepared_docs = [self._sanitize_document(doc) for doc in documents]
        batches = self._build_safe_batches(prepared_docs, action=action)

        url = f"{self.endpoint}/indexes/{index_name}/docs/index?api-version={self.api_version}"

        for batch_idx, payload in enumerate(batches, start=1):
            batch_uploaded, batch_failed = self._upload_payload_with_auto_split(
                url=url,
                payload=payload,
            )
            total_uploaded += batch_uploaded
            failed_items.extend(batch_failed)

            if batch_idx < len(batches):
                time.sleep(self.sleep_between_batches_sec)

        return {
            "status": "completed" if not failed_items else "partial_failure",
            "uploaded": total_uploaded,
            "failed": len(failed_items),
            "failed_items": failed_items[:20],
        }

    def prepare_and_index_documents(
        self,
        normalized_document: Dict[str, Any],
        index_documents: List[Dict[str, Any]],
        *,
        add_vectors: bool = True,
        embedder: Optional[AzureOpenAIEmbedder] = None,
    ) -> Dict[str, Any]:
        document_type = (
            normalized_document.get("document_type")
            or normalized_document.get("metadata", {}).get("document_type")
            or ""
        )
        document_id = (
            normalized_document.get("document_id")
            or normalized_document.get("metadata", {}).get("doc_id")
        )

        index_name = self.choose_index_name(document_type=document_type)

        create_result = self.create_index_if_missing(index_name)

        docs_to_upload = [dict(doc) for doc in index_documents]

        if add_vectors:
            embedder = embedder or AzureOpenAIEmbedder(dimensions=self.vector_dimensions)
            try:
                docs_to_upload = embedder.embed_documents(
                    docs_to_upload,
                    content_field=self.content_field,
                    vector_field=self.vector_field,
                )
            except EmbeddingError as exc:
                raise SearchIndexerError(f"Embedding failed before indexing: {exc}")

        upload_result = self.index_documents(index_name, docs_to_upload)

        return {
            "index_name": index_name,
            "document_type": document_type,
            "document_id": document_id,
            "chunk_count": len(index_documents),
            "vectorized": add_vectors,
            "index_create_result": create_result,
            "result": upload_result,
        }

    def _sanitize_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean values so Azure Search receives plain JSON serializable data.
        """
        clean: Dict[str, Any] = {}

        for key, value in doc.items():
            if value is None:
                clean[key] = None
                continue

            if key == self.vector_field and isinstance(value, list):
                try:
                    clean[key] = [float(x) for x in value]
                except Exception as exc:
                    raise SearchIndexerError(f"Invalid vector data for document {doc.get(self.id_field)}: {exc}")
                continue

            if isinstance(value, (str, int, float, bool, list, dict)):
                clean[key] = value
            else:
                clean[key] = str(value)

        return clean

    def _build_safe_batches(
        self,
        docs: List[Dict[str, Any]],
        *,
        action: str,
    ) -> List[Dict[str, Any]]:
        """
        Build batches constrained by:
        1. document count
        2. serialized payload byte size
        """
        batches: List[Dict[str, Any]] = []
        current_actions: List[Dict[str, Any]] = []
        current_size = self._payload_size_bytes({"value": []})

        for doc in docs:
            action_doc = {
                "@search.action": action,
                **doc,
            }

            action_doc_size = self._payload_size_bytes({"value": [action_doc]})

            # If one document alone is too large, still send it alone and let recursive split handler report clearly
            if not current_actions:
                current_actions.append(action_doc)
                current_size = self._payload_size_bytes({"value": current_actions})
                continue

            would_exceed_count = len(current_actions) >= self.batch_size
            would_exceed_bytes = (current_size + action_doc_size) > self.max_batch_bytes

            if would_exceed_count or would_exceed_bytes:
                batches.append({"value": current_actions})
                current_actions = [action_doc]
                current_size = self._payload_size_bytes({"value": current_actions})
            else:
                current_actions.append(action_doc)
                current_size += action_doc_size

        if current_actions:
            batches.append({"value": current_actions})

        return batches

    def _upload_payload_with_auto_split(
        self,
        *,
        url: str,
        payload: Dict[str, Any],
    ) -> (int, List[Dict[str, Any]]):
        """
        Upload payload.
        If a batch fails due to connection or payload size style issues,
        split it into smaller batches automatically.
        """
        actions = payload.get("value", []) or []
        if not actions:
            return 0, []

        try:
            result = self._request_with_retry("POST", url, json_body=payload)

            uploaded = 0
            failed_items: List[Dict[str, Any]] = []

            for item in result.get("value", []):
                if item.get("status") is True:
                    uploaded += 1
                else:
                    failed_items.append(item)

            return uploaded, failed_items

        except SearchIndexerError as exc:
            transient_like = self._looks_like_split_worth_error(exc)

            if len(actions) == 1 or not transient_like:
                doc_id = actions[0].get(self.id_field) if actions else None
                return 0, [
                    {
                        "key": doc_id,
                        "status": False,
                        "errorMessage": str(exc),
                    }
                ]

            mid = max(1, len(actions) // 2)
            left_payload = {"value": actions[:mid]}
            right_payload = {"value": actions[mid:]}

            left_uploaded, left_failed = self._upload_payload_with_auto_split(
                url=url,
                payload=left_payload,
            )
            time.sleep(self.sleep_between_batches_sec)
            right_uploaded, right_failed = self._upload_payload_with_auto_split(
                url=url,
                payload=right_payload,
            )

            return left_uploaded + right_uploaded, left_failed + right_failed

    def _payload_size_bytes(self, payload: Dict[str, Any]) -> int:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return len(raw.encode("utf-8"))

    def _looks_like_split_worth_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        split_signals = [
            "remote end closed connection without response",
            "remotedisconnected",
            "connection aborted",
            "connection reset",
            "timed out",
            "timeout",
            "413",
            "request entity too large",
            "transient search error 429",
            "transient search error 500",
            "transient search error 502",
            "transient search error 503",
            "transient search error 504",
        ]
        return any(signal in msg for signal in split_signals)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_body,
                    timeout=self.timeout_sec,
                )

                if response.status_code in (200, 201):
                    if response.text.strip():
                        return response.json()
                    return {}

                if response.status_code in (408, 413, 429, 500, 502, 503, 504):
                    raise SearchIndexerError(
                        f"Transient search error {response.status_code}: {response.text}"
                    )

                raise SearchIndexerError(
                    f"Search request failed {response.status_code}: {response.text}"
                )

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                RemoteDisconnected,
                SearchIndexerError,
            ) as exc:
                last_error = exc

                if attempt == self.max_retries:
                    break

                sleep_sec = min(2 ** (attempt - 1), 20)
                time.sleep(sleep_sec)

            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                sleep_sec = min(2 ** (attempt - 1), 20)
                time.sleep(sleep_sec)

        raise SearchIndexerError(f"Search request failed after retries: {last_error}")