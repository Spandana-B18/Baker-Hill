import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def env_get(name: str, default: str = "") -> str:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v


# ============================================================
# Azure OpenAI
# ============================================================
AZURE_OPENAI_ENDPOINT = env_get("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_OPENAI_KEY = env_get("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = env_get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env_get("AZURE_OPENAI_API_VERSION", "2025-04-14")

# ============================================================
# Content Understanding
# ============================================================
AZURE_CONTENT_UNDERSTANDING_ENDPOINT = env_get("AZURE_CONTENT_UNDERSTANDING_ENDPOINT").rstrip("/")
AZURE_CONTENT_UNDERSTANDING_KEY = env_get("AZURE_CONTENT_UNDERSTANDING_KEY")
CONTENT_UNDERSTANDING_ANALYZER_ID = env_get("CONTENT_UNDERSTANDING_ANALYZER_ID", "prebuilt-layout")

# ============================================================
# Azure Blob
# ============================================================
AZURE_STORAGE_CONNECTION_STRING = env_get("AZURE_STORAGE_CONNECTION_STRING")
BLOB_INPUT_CONTAINER = env_get("BLOB_INPUT_CONTAINER", "input-documents")
BLOB_OUTPUT_CONTAINER = env_get("BLOB_OUTPUT_CONTAINER", "output-json")
BLOB_LOG_CONTAINER = env_get("BLOB_LOG_CONTAINER", "logfiles")


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")