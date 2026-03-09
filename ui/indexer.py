import os
import time
from typing import Any, Dict, List, Optional

import requests

from embeddings import AzureOpenAIEmbedder, EmbeddingError


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

        self.batch_size = int(os.getenv("AZURE_SEARCH_BATCH_SIZE", "500"))
        self.max_retries = int(os.getenv("AZURE_SEARCH_MAX_RETRIES", "5"))
        self.timeout_sec = int(os.getenv("AZURE_SEARCH_TIMEOUT_SEC", "120"))

        if not self.endpoint:
            raise SearchIndexerError("Missing AZURE_SEARCH_SERVICE_ENDPOINT")
        if not self.admin_key:
            raise SearchIndexerError("Missing AZURE_SEARCH_ADMIN_KEY")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "api-key": self.admin_key,
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

        for batch in self._chunk_documents(documents, self.batch_size):
            payload = {
                "value": []
            }

            for doc in batch:
                payload["value"].append(
                    {
                        "@search.action": action,
                        **doc,
                    }
                )

            url = f"{self.endpoint}/indexes/{index_name}/docs/index?api-version={self.api_version}"
            result = self._request_with_retry("POST", url, json_body=payload)

            for item in result.get("value", []):
                if item.get("status") is True:
                    total_uploaded += 1
                else:
                    failed_items.append(item)

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
        document_type = normalized_document.get("document_type")
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
            "document_id": normalized_document.get("document_id"),
            "chunk_count": len(index_documents),
            "vectorized": add_vectors,
            "index_create_result": create_result,
            "result": upload_result,
        }

    def _chunk_documents(self, docs: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
        return [docs[i:i + size] for i in range(0, len(docs), size)]

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

                if response.status_code in (429, 500, 502, 503, 504):
                    raise SearchIndexerError(
                        f"Transient search error {response.status_code}: {response.text}"
                    )

                raise SearchIndexerError(
                    f"Search request failed {response.status_code}: {response.text}"
                )

            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                sleep_sec = min(2 ** (attempt - 1), 16)
                time.sleep(sleep_sec)

        raise SearchIndexerError(f"Search request failed after retries: {last_error}")