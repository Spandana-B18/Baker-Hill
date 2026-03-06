client = ContentUnderstandingClient(
    endpoint=AZURE_CONTENT_UNDERSTANDING_ENDPOINT,
    credential=AzureKeyCredential(AZURE_CONTENT_UNDERSTANDING_KEY),
)

# Try listing analyzers if supported by your SDK
try:
    analyzers = list(client.list_analyzers())
    st.write([a.analyzer_id for a in analyzers])
except Exception as e:
    st.write("list_analyzers not available or failed:", e)