import os
import io
import re
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import requests
from pypdf import PdfReader

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# -----------------------
# Config (.env)
# -----------------------
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AOAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14")
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AOAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

DI_ENDPOINT = os.getenv("AZURE_DI_ENDPOINT", "")
DI_API_VERSION = os.getenv("DI_API_VERSION", "2024-11-30")
DI_KEY = os.getenv("AZURE_DI_KEY", "")

PAGES_PER_BATCH = int(os.getenv("PAGES_PER_BATCH", "6"))
MAX_CHARS_PER_BATCH = int(os.getenv("MAX_CHARS_PER_BATCH", "18000"))


SYSTEM_PROMPT = """You extract factual key:value information from financial statement text.

Rules:
- Output JSON ONLY. No markdown.
- Do not use or assume any predefined financial schema.
- Invent keys as needed based on the text.
- Do not hallucinate: only include facts directly supported by the provided text.
- Every item must include:
  key, value, value_type, unit, period_or_asof, page, evidence_snippet, confidence
- evidence_snippet must be copied from the provided text (10-40 words).
- confidence must be 0..1.
"""


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Financial KV Extractor (Minimal)")

# Allow Streamlit (or any UI) to call this API locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/extract_kv")
async def extract_kv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a PDF -> returns JSON only:
    { source_file, page_count, kv:[...] }
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # 1) Extract per-page text (try digital first)
    pages = extract_pages_text_pdf(pdf_bytes)
    probe_chars = sum(len((p["text"] or "").strip()) for p in pages[:3])

    # 2) If looks scanned, OCR with DI Read (text only)
    if probe_chars < 200:
        pages = ocr_pages_text_di(pdf_bytes)

    # 3) Batch pages, call LLM per batch
    kv_all: List[Dict[str, Any]] = []
    for batch in split_pages_into_batches(pages, PAGES_PER_BATCH, MAX_CHARS_PER_BATCH):
        kv_all.extend(llm_extract_kv_from_text(batch["text"], batch["page_start"], batch["page_end"]))

    # 4) Dedupe
    final_kv = dedupe_kv(kv_all)

    return {
        "source_file": file.filename,
        "page_count": len(pages),
        "kv": final_kv
    }


# =========================================================
# Text extraction
# =========================================================
def extract_pages_text_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Digital PDF extraction using pypdf."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        out.append({"page": i + 1, "text": normalize_text(text)})
    return out


def ocr_pages_text_di(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Scanned PDF OCR using Document Intelligence prebuilt-read (text only)."""
    if not DI_ENDPOINT or not DI_KEY:
        raise HTTPException(status_code=500, detail="DI_ENDPOINT/DI_KEY not set for OCR.")

    submit_url = f"{DI_ENDPOINT}/documentintelligence/documentModels/prebuilt-read:analyze"
    params = {"api-version": DI_API_VERSION}
    headers = {"Ocp-Apim-Subscription-Key": DI_KEY, "Content-Type": "application/octet-stream"}

    resp = requests.post(submit_url, params=params, headers=headers, data=file_bytes, timeout=120)
    resp.raise_for_status()

    op = resp.headers.get("operation-location")
    if not op:
        raise HTTPException(status_code=500, detail="DI did not return operation-location.")

    for _ in range(120):  # ~4 minutes
        poll = requests.get(op, headers={"Ocp-Apim-Subscription-Key": DI_KEY}, timeout=60)
        poll.raise_for_status()
        payload = poll.json()
        status = payload.get("status")

        if status == "succeeded":
            return di_result_to_pages(payload.get("analyzeResult", {}))
        if status == "failed":
            raise HTTPException(status_code=500, detail="DI OCR failed.")

        time.sleep(2)

    raise HTTPException(status_code=504, detail="DI OCR timed out.")


def di_result_to_pages(analyze_result: dict) -> List[Dict[str, Any]]:
    pages = []
    for p in analyze_result.get("pages", []):
        page_num = int(p.get("pageNumber", 1))
        lines = p.get("lines", [])
        text = "\n".join(ln.get("content", "") for ln in lines)
        pages.append({"page": page_num, "text": normalize_text(text)})

    if not pages:
        pages = [{"page": 1, "text": normalize_text(analyze_result.get("content", ""))}]
    return pages


def normalize_text(text: str) -> str:
    """Light cleanup only."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+$", "", ln) for ln in text.split("\n"))
    return text


# =========================================================
# Batching (so 200 pages won't blow token limits)
# =========================================================
def split_pages_into_batches(pages: List[Dict[str, Any]], pages_per_batch: int, max_chars: int):
    cur = []
    start_page = None
    last_page = None

    for p in pages:
        page_num = p["page"]
        page_text = f"\n\n=== PAGE {page_num} ===\n{p.get('text','')}"
        if start_page is None:
            start_page = page_num

        candidate = "".join(cur) + page_text
        too_many_pages = (page_num - start_page + 1) > pages_per_batch
        too_many_chars = len(candidate) > max_chars

        if cur and (too_many_pages or too_many_chars):
            yield {"page_start": start_page, "page_end": last_page, "text": "".join(cur)}
            cur = []
            start_page = page_num

        cur.append(page_text)
        last_page = page_num

    if cur:
        yield {"page_start": start_page, "page_end": last_page, "text": "".join(cur)}


# =========================================================
# Azure OpenAI call (JSON-only output)
# =========================================================
def aoai_chat_json(system_prompt: str, user_prompt: str) -> dict:
    if not AOAI_ENDPOINT or not AOAI_KEY:
        raise HTTPException(status_code=500, detail="AOAI_ENDPOINT/AOAI_KEY not set.")

    url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_CHAT_DEPLOYMENT}/chat/completions"
    params = {"api-version": AOAI_API_VERSION}
    headers = {"api-key": AOAI_KEY, "Content-Type": "application/json"}
    payload = {
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(url, params=params, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def llm_extract_kv_from_text(batch_text: str, page_start: int, page_end: int) -> List[Dict[str, Any]]:
    user_prompt = f"""Extract as many supported key-value facts as possible from this text (pages {page_start}-{page_end}).

Return JSON:
{{
  "items": [
    {{
      "key": "Some.Invented.Key.2025",
      "value": 1234.0,
      "value_type": "number|string|date|boolean|null",
      "unit": "USD|USD_millions|USD_thousands|percent|shares|null",
      "period_or_asof": "FY2025|2025-12-31|null",
      "page": {page_start},
      "evidence_snippet": "copied excerpt",
      "confidence": 0.0
    }}
  ]
}}

Text:
{batch_text}
"""
    out = aoai_chat_json(SYSTEM_PROMPT, user_prompt)
    items = out.get("items", [])
    if not isinstance(items, list):
        return []
    return [coerce_item(it, page_start, page_end) for it in items if isinstance(it, dict)]


def coerce_item(it: dict, page_start: int, page_end: int) -> dict:
    try:
        page = int(it.get("page", page_start))
    except Exception:
        page = page_start
    if page < page_start or page > page_end:
        page = page_start

    try:
        conf = float(it.get("confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    return {
        "key": it.get("key"),
        "value": it.get("value", None),
        "value_type": it.get("value_type", "null"),
        "unit": it.get("unit", None),
        "period_or_asof": it.get("period_or_asof", None),
        "page": page,
        "evidence_snippet": it.get("evidence_snippet", ""),
        "confidence": conf,
    }


# =========================================================
# Simple dedupe
# =========================================================
def dedupe_kv(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for it in items:
        key = normalize_key(it.get("key"))
        if not key:
            continue
        it["key"] = key
        prev = best.get(key)
        if prev is None or float(it.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            best[key] = it

    out = list(best.values())
    out.sort(key=lambda x: (x.get("page") or 0, x.get("key") or ""))
    return out


def normalize_key(k: Any) -> str:
    if not isinstance(k, str):
        return ""
    return re.sub(r"\s+", "", k.strip())