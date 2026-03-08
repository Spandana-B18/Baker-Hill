import os
import uuid
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# ----------------------------------------------------------
# Load environment variables
# ----------------------------------------------------------
load_dotenv()

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")

FINANCIAL_INDEX = os.getenv("AZURE_SEARCH_FINANCIAL_INDEX")
TAX_INDEX = os.getenv("AZURE_SEARCH_TAX_INDEX")

CONTENT_FIELD = os.getenv("AZURE_SEARCH_CONTENT_FIELD", "content")
TITLE_FIELD = os.getenv("AZURE_SEARCH_TITLE_FIELD", "document_type")
ID_FIELD = os.getenv("AZURE_SEARCH_ID_FIELD", "doc_id")


# ----------------------------------------------------------
# Validate environment variables
# ----------------------------------------------------------
def validate_config():

    if not SEARCH_ENDPOINT:
        raise ValueError("AZURE_SEARCH_ENDPOINT is missing")

    if not SEARCH_KEY:
        raise ValueError("AZURE_SEARCH_KEY is missing")

    if not FINANCIAL_INDEX:
        raise ValueError("AZURE_SEARCH_FINANCIAL_INDEX is missing")

    if not TAX_INDEX:
        raise ValueError("AZURE_SEARCH_TAX_INDEX is missing")


# ----------------------------------------------------------
# Flatten LLM payload
# Example:
# payload:
#   revenue:
#       value: 50000
#       confidence_score: 0.92
#
# becomes:
# revenue: 50000
# ----------------------------------------------------------
def flatten_payload(payload):

    flattened = {}

    if not payload:
        return flattened

    for field, data in payload.items():

        if isinstance(data, dict):
            flattened[field] = data.get("value")

        else:
            flattened[field] = data

    return flattened


# ----------------------------------------------------------
# Build Search Client
# ----------------------------------------------------------
def get_search_client(index_name):

    validate_config()

    return SearchClient(
        endpoint=SEARCH_ENDPOINT.strip(),
        index_name=index_name,
        credential=AzureKeyCredential(SEARCH_KEY.strip())
    )


# ----------------------------------------------------------
# Determine index based on schema
# ----------------------------------------------------------
def get_index_name(schema_id):

    if not schema_id:
        return FINANCIAL_INDEX

    schema_lower = schema_id.lower()

    if "tax" in schema_lower:
        return TAX_INDEX

    return FINANCIAL_INDEX


# ----------------------------------------------------------
# Upload document to Azure AI Search
# ----------------------------------------------------------
def upload_document_to_search(extracted_json):

    try:

        metadata = extracted_json.get("metadata", {})
        payload = extracted_json.get("payload", {})
        schema = extracted_json.get("schema", {})

        schema_id = schema.get("schema_id", "unknown_document")

        # Flatten payload
        flattened = flatten_payload(payload)

        # Ensure document ID exists
        doc_id = metadata.get("doc_id") or str(uuid.uuid4())

        # Build searchable content
        content_text = " ".join(
            [f"{k}:{v}" for k, v in flattened.items() if v is not None]
        )

        # Build document
        document = {
            ID_FIELD: doc_id,
            TITLE_FIELD: schema_id,
            CONTENT_FIELD: content_text,
            **flattened
        }

        # Select index
        index_name = get_index_name(schema_id)

        print(f"Uploading document to index: {index_name}")
        print(f"Document ID: {doc_id}")

        # Upload
        client = get_search_client(index_name)

        result = client.upload_documents(documents=[document])

        print("Azure Search Upload Result:", result)

        return result

    except Exception as e:

        print("Azure Search Upload Failed:", str(e))
        raise