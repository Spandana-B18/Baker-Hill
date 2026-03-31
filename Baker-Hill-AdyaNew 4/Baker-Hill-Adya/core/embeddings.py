import os
import time
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


class EmbeddingError(Exception):
    pass


def _chunk_list(items: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


class AzureOpenAIEmbedder:
    """
    Uses Azure OpenAI v1 embeddings endpoint.

    Required env vars:
      AZURE_OPENAI_ENDPOINT
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_EMBEDDING_DEPLOYMENT

    Optional env vars:
      AZURE_OPENAI_EMBEDDING_DIMENSIONS
      AZURE_OPENAI_EMBEDDING_BATCH_SIZE
      AZURE_OPENAI_EMBEDDING_MAX_RETRIES
      AZURE_OPENAI_EMBEDDING_TIMEOUT_SEC
    """

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        self.endpoint = (endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")).rstrip("/")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

        dims_raw = os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "").strip()
        self.dimensions = dimensions if dimensions is not None else (int(dims_raw) if dims_raw else None)

        self.batch_size = batch_size or int(os.getenv("AZURE_OPENAI_EMBEDDING_BATCH_SIZE", "64"))
        self.max_retries = max_retries or int(os.getenv("AZURE_OPENAI_EMBEDDING_MAX_RETRIES", "5"))
        self.timeout_sec = int(os.getenv("AZURE_OPENAI_EMBEDDING_TIMEOUT_SEC", "120"))

        if not self.endpoint:
            raise EmbeddingError("Missing AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            raise EmbeddingError("Missing AZURE_OPENAI_API_KEY")
        if not self.deployment:
            raise EmbeddingError("Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.endpoint}/openai/v1/",
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_vectors: List[List[float]] = []

        for batch in _chunk_list(texts, self.batch_size):
            response = self._embed_batch_with_retry(batch)

            if not hasattr(response, "data") or response.data is None:
                raise EmbeddingError("Embedding response missing data")

            batch_vectors: List[List[float]] = []
            for item in response.data:
                vector = getattr(item, "embedding", None)
                if not isinstance(vector, list) or not vector:
                    raise EmbeddingError("Embedding response contained an empty vector")
                if self.dimensions is not None and len(vector) != self.dimensions:
                    raise EmbeddingError(
                        f"Embedding dimension mismatch. Expected {self.dimensions}, got {len(vector)}"
                    )
                batch_vectors.append(vector)

            if len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    f"Embedding count mismatch. Expected {len(batch)}, got {len(batch_vectors)}"
                )

            all_vectors.extend(batch_vectors)

        return all_vectors

    def _embed_batch_with_retry(self, batch: List[str]):
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.deployment,
                    "input": batch,
                }

                if self.dimensions is not None:
                    kwargs["dimensions"] = self.dimensions

                return self.client.embeddings.create(**kwargs)

            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                sleep_sec = min(2 ** (attempt - 1), 16)
                time.sleep(sleep_sec)

        raise EmbeddingError(f"Failed to generate embeddings after retries: {last_error}")

    def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        *,
        content_field: str = "content",
        vector_field: str = "content_vector",
        skip_empty: bool = True,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        texts: List[str] = []
        doc_positions: List[int] = []

        for idx, doc in enumerate(documents):
            text = str(doc.get(content_field, "") or "").strip()
            if not text:
                if skip_empty:
                    continue
                raise EmbeddingError(f"Document at index {idx} has empty content")
            texts.append(text)
            doc_positions.append(idx)

        if not texts:
            return documents

        vectors = self.embed_texts(texts)

        for pos, vector in zip(doc_positions, vectors):
            documents[pos][vector_field] = vector

        return documents

