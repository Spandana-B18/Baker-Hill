import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL_DEFAULT = "http://localhost:8000"

st.set_page_config(page_title="Financial KV UI", layout="wide")
st.title("📄 Financial Statement → Key:Value Extractor")
st.caption("Frontend (Streamlit) → Backend (FastAPI). Output = JSON.")

with st.sidebar:
    backend_url = st.text_input("Backend URL", value=os.getenv("BACKEND_URL", BACKEND_URL_DEFAULT))
    timeout_s = st.number_input("Request timeout (seconds)", 30, 3600, 600)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
run = st.button("🚀 Extract JSON", type="primary", disabled=(uploaded is None))

if run and uploaded is not None:
    with st.spinner("Sending to backend…"):
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        try:
            resp = requests.post(f"{backend_url.rstrip('/')}/extract_kv", files=files, timeout=int(timeout_s))
        except requests.exceptions.Timeout:
            st.error("Timed out. Try lowering PAGES_PER_BATCH in .env.")
            st.stop()

    if resp.status_code >= 400:
        st.error(f"Backend error {resp.status_code}")
        st.code(resp.text)
        st.stop()

    data = resp.json()
    st.success("Done ✅")

    st.subheader("JSON Output")
    st.json(data)

    st.download_button(
        "Download result.json",
        data=json.dumps(data, indent=2),
        file_name="result.json",
        mime="application/json",
    )