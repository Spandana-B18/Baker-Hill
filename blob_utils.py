import json
from typing import List

from azure.storage.blob import BlobServiceClient


def ensure_container(blob_service: BlobServiceClient, name: str) -> None:
    try:
        blob_service.create_container(name)
    except Exception:
        pass


def blob_upload_bytes(
    blob_service: BlobServiceClient,
    container: str,
    blob_name: str,
    data: bytes,
    content_type: str,
) -> None:
    bc = blob_service.get_blob_client(container=container, blob=blob_name)
    bc.upload_blob(data, overwrite=True, content_type=content_type)


def blob_upload_json(blob_service: BlobServiceClient, container: str, blob_name: str, obj: dict) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    blob_upload_bytes(blob_service, container, blob_name, data, "application/json")


def list_blobs(blob_service: BlobServiceClient, container: str, suffix: str = ".pdf") -> List[str]:
    """List blob names in a container, optionally filtered by suffix (e.g. .pdf)."""
    try:
        container_client = blob_service.get_container_client(container)
        names = [b.name for b in container_client.list_blobs() if not suffix or b.name.lower().endswith(suffix)]
        return sorted(names)
    except Exception:
        return []


def blob_download_bytes(blob_service: BlobServiceClient, container: str, blob_name: str) -> bytes:
    """Download a blob's content as bytes."""
    client = blob_service.get_blob_client(container=container, blob=blob_name)
    return client.download_blob().readall()

