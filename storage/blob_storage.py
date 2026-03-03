import json
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
