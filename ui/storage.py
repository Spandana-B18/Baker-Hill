import json
import os
from typing import Any, Dict, Optional

from azure.storage.blob import BlobServiceClient, ContentSettings


def get_blob_service_client() -> BlobServiceClient:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()

    if not connection_string:
        raise ValueError("Missing AZURE_STORAGE_CONNECTION_STRING")

    return BlobServiceClient.from_connection_string(connection_string)


def get_container_client(container_name: str):
    service_client = get_blob_service_client()
    container_client = service_client.get_container_client(container_name)

    try:
        container_client.create_container()
    except Exception:
        pass

    return container_client


def upload_bytes_to_blob(
    container_name: str,
    blob_name: str,
    data: bytes,
    content_type: Optional[str] = None,
) -> str:
    container_client = get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    if content_type:
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    else:
        blob_client.upload_blob(data, overwrite=True)

    return blob_name


def upload_json_to_blob(
    container_name: str,
    blob_name: str,
    data: Dict[str, Any],
) -> str:
    payload = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    return upload_bytes_to_blob(
        container_name=container_name,
        blob_name=blob_name,
        data=payload,
        content_type="application/json",
    )


def download_blob_bytes(
    container_name: str,
    blob_name: str,
) -> bytes:
    container_client = get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def blob_exists(container_name: str, blob_name: str) -> bool:
    container_client = get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.exists()