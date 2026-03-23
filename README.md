# Baker Hill

A Streamlit application for document processing, extraction, and AI-powered question answering. Upload documents, run Azure Content Understanding for extraction, index them into Azure AI Search, and ask questions directly from the extracted content.

## Features

- **Upload and Index**: Upload PDF, PNG, JPG, TIFF, BMP, or HEIF files
- **Azure Content Understanding**: Extract text, tables, and structure from documents
- **Document Classification**: Automatic detection of tax, financial, or generic documents
- **Normalization**: Build structured JSON from raw extraction
- **Azure AI Search**: Generate embeddings and index chunks for semantic search
- **Ask Questions**: Query documents with AI using extracted JSON as context
- **Confidence Metrics**: View extraction quality (mean confidence, low-confidence line percentage)
- **Blob Storage**: Persist source files, raw JSON, normalized JSON, and run logs

## Project Structure

```
Baker-Hill/
├── ui/
│   ├── app.py              # Streamlit UI and pipeline orchestration
│   └── assets/
│   └── css/
│       └── logo.png        # Application logo (optional)
├── core/                   # Core processing and AI services
│   ├── conf_score.py       # Content Understanding, confidence scoring, document schemas
│   ├── doc_qa.py           # Document Q&A (direct JSON context)
│   ├── embeddings.py       # Azure OpenAI embeddings
│   ├── indexer.py          # Azure AI Search indexing
│   └── retrieval_llm.py    # Retrieval and answer pipeline
├── ingest/                 # Ingest and transformation
│   ├── storage.py          # Azure Blob Storage helpers
│   └── transform.py        # Document normalization and chunk building
├── requirements.txt
└── .env                    # Environment variables (create from .env.example)
```

## Prerequisites

- Python 3.9+
- Azure subscription with:
  - **Azure AI Content Understanding** (document analysis)
  - **Azure Blob Storage** (input, output, logs)
  - **Azure AI Search** (vector indexing)
  - **Azure OpenAI** (embeddings and chat for Q&A)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Baker-Hill
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux/macOS
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Azure credentials (see [Environment Variables](#environment-variables)).

4. Place your logo (optional) at `ui/css/logo.png`.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_CONTENT_UNDERSTANDING_ENDPOINT` | Azure Content Understanding endpoint |
| `AZURE_CONTENT_UNDERSTANDING_KEY` | API key for Content Understanding |
| `AZURE_STORAGE_CONNECTION_STRING` | Blob storage connection string |
| `BLOB_INPUT_CONTAINER` | Container for uploaded source files (default: `input-documents`) |
| `BLOB_OUTPUT_CONTAINER` | Container for raw/normalized JSON (default: `output-json`) |
| `BLOB_LOG_CONTAINER` | Container for run logs (default: `logfiles`) |
| `AZURE_SEARCH_SERVICE_ENDPOINT` | Azure AI Search endpoint |
| `AZURE_SEARCH_ADMIN_KEY` | Admin key for AI Search |
| `AZURE_SEARCH_TAX_INDEX` | Index for tax documents (default: `tax-documents-index`) |
| `AZURE_SEARCH_FINANCIAL_INDEX` | Index for financial documents (default: `financial-documents-index`) |
| `AZURE_SEARCH_GENERIC_INDEX` | Index for generic documents (default: `generic-documents-index`) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | API key for Azure OpenAI |
| `AZURE_OPENAI_DEPLOYMENT` | Chat deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name |
| `MAX_FILE_SIZE_MB` | Maximum upload size in MB (default: `10`) |

## Run

From the project root:

```bash
streamlit run ui/app.py
```

Or from the `ui` folder:

```bash
cd ui
streamlit run app.py
```

The app runs at `http://localhost:8501` by default.

## Usage

1. **Upload and Index** tab:  
   - Upload a document (PDF or image)  
   - Click **Run analysis** to run the full pipeline: upload → extract → normalize → embed → index  
   - View raw/normalized JSON previews and download artifacts  

2. **Ask Questions** tab:  
   - After indexing, use this tab to ask natural-language questions about the document  
   - Answers are grounded in the extracted content  

## License

Proprietary. All rights reserved.

Final code
