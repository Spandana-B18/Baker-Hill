import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from .embeddings import AzureOpenAIEmbedder, EmbeddingError


class RetrievalError(Exception):
    pass


class AzureSearchRetriever:
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

        self.content_field = os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content")
        self.vector_field = os.getenv("AZURE_SEARCH_VECTOR_FIELD", "content_vector")
        self.timeout_sec = int(os.getenv("AZURE_SEARCH_TIMEOUT_SEC", "120"))
        self.max_retries = int(os.getenv("AZURE_SEARCH_MAX_RETRIES", "5"))

        if not self.endpoint:
            raise RetrievalError("Missing AZURE_SEARCH_SERVICE_ENDPOINT")
        if not self.admin_key:
            raise RetrievalError("Missing AZURE_SEARCH_ADMIN_KEY")

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

    def search_hybrid(
        self,
        *,
        query: str,
        index_name: str,
        query_vector: Optional[List[float]] = None,
        top: int = 5,
        filter_expression: Optional[str] = None,
        use_semantic: bool = False,
    ) -> Dict[str, Any]:
        url = f"{self.endpoint}/indexes/{index_name}/docs/search?api-version={self.api_version}"

        payload: Dict[str, Any] = {
            "count": True,
            "top": top,
            "select": ",".join(
                [
                    "id",
                    "document_id",
                    "chunk_id",
                    "title",
                    "content",
                    "document_type",
                    "document_subtype",
                    "schema_id",
                    "schema_version",
                    "source_file_name",
                    "source_blob",
                    "raw_json_blob",
                    "page_number",
                    "chunk_type",
                    "line_count",
                    "confidence_score",
                    "processed_at",
                ]
            ),
            "search": query,
        }

        if query_vector:
            payload["vectorQueries"] = [
                {
                    "kind": "vector",
                    "vector": query_vector,
                    "fields": self.vector_field,
                    "k": top,
                }
            ]

        if filter_expression:
            payload["filter"] = filter_expression

        if use_semantic:
            payload["queryType"] = "semantic"
            payload["semanticConfiguration"] = "default-semantic-config"

        return self._request_with_retry("POST", url, json_body=payload)

    def search_keyword_only(
        self,
        *,
        query: str,
        index_name: str,
        top: int = 5,
        filter_expression: Optional[str] = None,
        use_semantic: bool = False,
    ) -> Dict[str, Any]:
        return self.search_hybrid(
            query=query,
            index_name=index_name,
            query_vector=None,
            top=top,
            filter_expression=filter_expression,
            use_semantic=use_semantic,
        )

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
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504):
                    raise RetrievalError(
                        f"Transient Azure AI Search error {response.status_code}: {response.text}"
                    )

                raise RetrievalError(
                    f"Azure AI Search request failed {response.status_code}: {response.text}"
                )

            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 16))

        raise RetrievalError(f"Azure AI Search request failed after retries: {last_error}")


class AzureOpenAIAnswerer:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
        self.timeout_sec = int(os.getenv("LLM_TIMEOUT_SEC", "120"))

        if not self.endpoint:
            raise RetrievalError("Missing AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            raise RetrievalError("Missing AZURE_OPENAI_API_KEY")
        if not self.deployment:
            raise RetrievalError("Missing AZURE_OPENAI_DEPLOYMENT")
        if not self.api_version:
            raise RetrievalError("Missing AZURE_OPENAI_API_VERSION")

    def answer(
        self,
        *,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        context_blocks: List[Dict[str, Any]] = []

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                {
                    "rank": rank,
                    "document_id": chunk.get("document_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "title": chunk.get("title"),
                    "source_file_name": chunk.get("source_file_name"),
                    "page_number": chunk.get("page_number"),
                    "document_type": chunk.get("document_type"),
                    "content": chunk.get("content"),
                }
            )

        system_prompt = (
            "You are a grounded document QA assistant. "
            "Answer only from the provided retrieved chunks. "
            "Do not invent facts. "
            "If the answer is not supported by the retrieved chunks, say so clearly. "
            "Return valid JSON only with this structure: "
            '{"answer": <string>, "grounded": <true_or_false>, '
            '"citations": [{"rank": <int>, "source_file_name": <string_or_null>, '
            '"page_number": <int_or_null>, "chunk_id": <string_or_null>}]}.'  # noqa: E501
        )

        user_payload = {
            "question": query,
            "document_type": document_type,
            "retrieved_chunks": context_blocks,
        }

        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)

        if response.status_code != 200:
            raise RetrievalError(
                f"Azure OpenAI answer generation failed {response.status_code}: {response.text}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RetrievalError("Azure OpenAI returned no choices")

        content = (choices[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RetrievalError("Azure OpenAI returned empty content")

        try:
            parsed = json.loads(content)
        except Exception as exc:
            raise RetrievalError(f"Answer JSON parsing failed: {exc}")

        if not isinstance(parsed, dict):
            raise RetrievalError("Answer payload must be a JSON object")

        return parsed


class RetrievalPipeline:
    def __init__(self):
        self.retriever = AzureSearchRetriever()
        self.answerer = AzureOpenAIAnswerer()
        self.embedder = AzureOpenAIEmbedder()

    def route_query(self, query: str, forced_document_type: Optional[str] = None) -> str:
        if forced_document_type:
            return forced_document_type

        q = query.lower()

        tax_terms = [
            "tax",
            "irs",
            "ein",
            "1040",
            "1120",
            "1120-s",
            "k-1",
            "schedule l",
            "adjusted gross income",
        ]
        financial_terms = [
            "balance sheet",
            "income statement",
            "cash flow",
            "financial statement",
            "net income",
            "assets",
            "liabilities",
            "equity",
            "inventory",
            "accounts receivable",
        ]

        tax_score = sum(1 for term in tax_terms if term in q)
        financial_score = sum(1 for term in financial_terms if term in q)

        if tax_score > financial_score and tax_score > 0:
            return "tax_document"
        if financial_score > tax_score and financial_score > 0:
            return "financial_document"
        return "generic_document"

    def retrieve(
        self,
        *,
        query: str,
        forced_document_type: Optional[str] = None,
        top_k: int = 5,
        filter_expression: Optional[str] = None,
        use_semantic: bool = False,
    ) -> Dict[str, Any]:
        document_type = self.route_query(query, forced_document_type)
        index_name = self.retriever.choose_index_name(document_type)

        query_vector: Optional[List[float]] = None
        vector_mode_used = False

        try:
            query_vector = self.embedder.embed_texts([query])[0]
            vector_mode_used = True
        except EmbeddingError:
            query_vector = None
            vector_mode_used = False

        try:
            if query_vector is not None:
                search_response = self.retriever.search_hybrid(
                    query=query,
                    index_name=index_name,
                    query_vector=query_vector,
                    top=top_k,
                    filter_expression=filter_expression,
                    use_semantic=use_semantic,
                )
            else:
                search_response = self.retriever.search_keyword_only(
                    query=query,
                    index_name=index_name,
                    top=top_k,
                    filter_expression=filter_expression,
                    use_semantic=use_semantic,
                )
        except RetrievalError:
            if vector_mode_used:
                search_response = self.retriever.search_keyword_only(
                    query=query,
                    index_name=index_name,
                    top=top_k,
                    filter_expression=filter_expression,
                    use_semantic=use_semantic,
                )
                vector_mode_used = False
            else:
                raise

        results = search_response.get("value", []) or []

        return {
            "query": query,
            "document_type": document_type,
            "index_name": index_name,
            "vector_mode_used": vector_mode_used,
            "result_count": len(results),
            "results": results,
            "raw_search_response": search_response,
        }

    def answer(
        self,
        *,
        query: str,
        forced_document_type: Optional[str] = None,
        top_k: int = 5,
        filter_expression: Optional[str] = None,
        use_semantic: bool = False,
    ) -> Dict[str, Any]:
        retrieval = self.retrieve(
            query=query,
            forced_document_type=forced_document_type,
            top_k=top_k,
            filter_expression=filter_expression,
            use_semantic=use_semantic,
        )

        results = retrieval["results"]

        if not results:
            return {
                "query": query,
                "document_type": retrieval["document_type"],
                "index_name": retrieval["index_name"],
                "vector_mode_used": retrieval["vector_mode_used"],
                "answer": "I could not find relevant indexed content for this question.",
                "grounded": False,
                "citations": [],
                "results": [],
            }

        answer_payload = self.answerer.answer(
            query=query,
            retrieved_chunks=results,
            document_type=retrieval["document_type"],
        )

        return {
            "query": query,
            "document_type": retrieval["document_type"],
            "index_name": retrieval["index_name"],
            "vector_mode_used": retrieval["vector_mode_used"],
            "answer": answer_payload.get("answer"),
            "grounded": answer_payload.get("grounded"),
            "citations": answer_payload.get("citations", []),
            "results": results,
        }


def answer_user_query(
    query: str,
    *,
    forced_document_type: Optional[str] = None,
    top_k: int = 5,
    filter_expression: Optional[str] = None,
    use_semantic: bool = False,
) -> Dict[str, Any]:
    pipeline = RetrievalPipeline()
    return pipeline.answer(
        query=query,
        forced_document_type=forced_document_type,
        top_k=top_k,
        filter_expression=filter_expression,
        use_semantic=use_semantic,
    )

