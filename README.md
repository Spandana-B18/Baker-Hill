# Baker Hill POC

Streamlit app that:
1. Uploads a PDF to Azure Blob Storage
2. Runs Azure AI Content Understanding to get markdown, pages, and tables
3. Uses Azure OpenAI to extract a JSON envelope in chunks
4. Runs optional LLM business validation and deterministic schema validation
5. Writes logs and final JSON back to Blob Storage

## Project layout

- app/streamlit_app.py
  Streamlit UI and pipeline orchestration

- baker_hill_poc/
  Python package with config, services, schemas, storage helpers

## Run

1. Create and fill a `.env` file using `.env.example`
2. Install dependencies
3. Start Streamlit

Example:

python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
