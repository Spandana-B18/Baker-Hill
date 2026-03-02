import os
import io
import re
import json
import base64
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import requests
from pypdf import PdfReader
import fitz  # PyMuPDF
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# =========================================================
# Load environment variables
# =========================================================
load_dotenv()


# =========================================================
# Config (.env)
# =========================================================
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AOAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-14")
AOAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AOAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# DIGITAL text batching
PAGES_PER_BATCH_TEXT = int(os.getenv("PAGES_PER_BATCH_TEXT", "10"))
MAX_CHARS_PER_BATCH = int(os.getenv("MAX_CHARS_PER_BATCH", "28000"))

# SCANNED vision batching
PAGES_PER_BATCH_IMAGE = int(os.getenv("PAGES_PER_BATCH_IMAGE", "2"))
MAX_IMAGE_PAGES = int(os.getenv("MAX_IMAGE_PAGES", "40"))
RENDER_DPI = int(os.getenv("RENDER_DPI", "170"))

MAX_LLM_CONCURRENCY = int(os.getenv("MAX_LLM_CONCURRENCY", "3"))

# scan detection threshold
SCAN_PROBE_CHAR_THRESHOLD = int(os.getenv("SCAN_PROBE_CHAR_THRESHOLD", "200"))


# =========================================================
# Prompt: nested, analyst-style JSON
# =========================================================
SYSTEM_PROMPT = """You are a senior financial statement analyst with many years of experience 
reviewing financial statements and extracting decision-useful information.

Your task:
Extract ALL relevant financial facts from the provided content and return them as JSON ONLY.

Important:
- The output MUST be a single JSON object.
- You MAY nest objects where it improves clarity (recommended).
- Use human-readable keys (no dot-keys).

Preferred nesting (use if the content supports it):
{
  "Reporting": {
    "Company Name": "...",
    "Document Type": "Annual Report / Financial Statements / etc.",
    "Reporting Period": "FY 2024",
    "As of Date": "December 31, 2024",
    "Currency / Units": "USD (in millions)"
  },
  "Income Statement": { ... },
  "Balance Sheet": { ... },
  "Cash Flow Statement": { ... },
  "Notes / Disclosures": {
    "Debt": { ... },
    "Leases": { ... },
    "Commitments / Contingencies": { ... }
  }
}

Rules:
- JSON ONLY. No markdown. No explanation.
- No predefined schema: invent sections/keys as needed.
- Keys must be human-readable and descriptive, e.g.:
  "Total Revenue (FY 2024, USD millions)": 1250.4
  "Total Assets (As of December 31, 2024, USD thousands)": 5400.6
- Include period / as-of date AND currency/units in the key whenever present.
- Only include facts explicitly stated in the content. Do NOT hallucinate.
- If a value is unclear or not explicitly present, omit it.
"""


# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(title="Financial Statement Extractor (Vision, Nested JSON, No DI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract_kv")
async def extract_kv(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # 1) Try digital extraction
    pages_text = extract_pages_text_pdf(pdf_bytes)
    probe_chars = sum(len((p["text"] or "").strip()) for p in pages_text[:3])
    looks_scanned = probe_chars < SCAN_PROBE_CHAR_THRESHOLD

    if not looks_scanned:
        # DIGITAL path: extract text -> batch -> parallel LLM -> deep merge
        pages_text = [p for p in pages_text if is_useful_page(p["text"])]
        batches = list(split_pages_into_text_batches(pages_text, PAGES_PER_BATCH_TEXT, MAX_CHARS_PER_BATCH))
        extracted = await extract_text_batches_parallel(batches)

        return build_response_with_filename_first(file.filename, extracted)

    # SCANNED path: render to images -> vision LLM -> deep merge
    page_images = render_pdf_to_png_dataurls(pdf_bytes, max_pages=MAX_IMAGE_PAGES, dpi=RENDER_DPI)
    image_batches = split_list(page_images, PAGES_PER_BATCH_IMAGE)
    extracted = await extract_image_batches_parallel(image_batches)

    # Add a small note in the response (optional). Remove if you don't want it.
    if isinstance(extracted, dict):
        extracted.setdefault("Processing Notes", {})
        if isinstance(extracted["Processing Notes"], dict):
            extracted["Processing Notes"].setdefault(
                "Scanned PDF Limit",
                f"Processed up to {MAX_IMAGE_PAGES} pages for scanned PDFs."
            )

    return build_response_with_filename_first(file.filename, extracted)


# =========================================================
# Response shape: filename first
# =========================================================
def build_response_with_filename_first(filename: str, extracted: Any) -> Dict[str, Any]:
    """
    Returns a JSON object whose first inserted key is "source_file".
    Note: JSON key order isn't guaranteed by the spec, but Python/FastAPI typically preserves insertion order.
    """
    safe_extracted = extracted if isinstance(extracted, dict) else {}
    # Prevent collisions
    if "source_file" in safe_extracted:
        safe_extracted = {k: v for k, v in safe_extracted.items() if k != "source_file"}

    ordered = OrderedDict()
    ordered["source_file"] = filename
    for k, v in safe_extracted.items():
        ordered[k] = v
    return ordered


# =========================================================
# Digital PDF text extraction
# =========================================================
def extract_pages_text_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": normalize_text(text)})
    return pages


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def is_useful_page(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 80:
        return False
    alpha = sum(c.isalnum() for c in t)
    return (alpha / max(1, len(t))) > 0.15


def split_pages_into_text_batches(pages: List[Dict[str, Any]], pages_per_batch: int, max_chars: int):
    cur = []
    start_page = None
    last_page = None

    for p in pages:
        page_num = p["page"]
        page_text = f"\n\n=== PAGE {page_num} ===\n{p['text']}"

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
# Scanned PDF -> images (data URLs)
# =========================================================
def render_pdf_to_png_dataurls(pdf_bytes: bytes, max_pages: int = 40, dpi: int = 170) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    page_count = min(len(doc), max_pages)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    out: List[str] = []
    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        out.append(f"data:image/png;base64,{b64}")

    return out


def split_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# =========================================================
# Parallel extraction + deep merge
# =========================================================
async def extract_text_batches_parallel(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    sem = asyncio.Semaphore(MAX_LLM_CONCURRENCY)
    executor = ThreadPoolExecutor(MAX_LLM_CONCURRENCY)

    async def run_one(batch: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                executor,
                llm_extract_nested_json_from_text_batch,
                batch["text"],
                batch["page_start"],
                batch["page_end"],
            )

    results = await asyncio.gather(*(run_one(b) for b in batches))
    return deep_merge_many(results)


async def extract_image_batches_parallel(image_batches: List[List[str]]) -> Dict[str, Any]:
    sem = asyncio.Semaphore(MAX_LLM_CONCURRENCY)
    executor = ThreadPoolExecutor(MAX_LLM_CONCURRENCY)

    async def run_one(images: List[str], batch_index: int) -> Dict[str, Any]:
        async with sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                executor,
                llm_extract_nested_json_from_image_batch,
                images,
                batch_index,
            )

    results = await asyncio.gather(*(run_one(imgs, idx) for idx, imgs in enumerate(image_batches, start=1)))
    return deep_merge_many(results)


def deep_merge_many(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for d in dicts:
        if isinstance(d, dict):
            merged = deep_merge(merged, d)
    return merged


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive merge:
    - dict + dict => merge keys
    - list + list => append unique-ish (simple)
    - otherwise => b overwrites a
    """
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
            continue

        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        elif isinstance(out[k], list) and isinstance(v, list):
            out[k] = out[k] + [x for x in v if x not in out[k]]
        else:
            out[k] = v
    return out


# =========================================================
# Azure OpenAI calls (Vision + JSON object)
# =========================================================
def aoai_chat_json(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not AOAI_ENDPOINT or not AOAI_KEY:
        raise HTTPException(status_code=500, detail="Azure OpenAI env vars not set.")

    url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_CHAT_DEPLOYMENT}/chat/completions"
    payload = {
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(
        url,
        params={"api-version": AOAI_API_VERSION},
        headers={"api-key": AOAI_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=240,
    )
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def llm_extract_nested_json_from_text_batch(batch_text: str, page_start: int, page_end: int) -> Dict[str, Any]:
    user_prompt = f"""Extract all relevant financial information from pages {page_start}-{page_end}.

Return JSON ONLY as a single JSON object.
- Use human-readable keys
- Nest where helpful (Income Statement / Balance Sheet / Cash Flow / Notes)
- Only include facts explicitly supported by the text / document (no guessing)
- Omit unclear values

Text:
{batch_text}
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return aoai_chat_json(messages)


def llm_extract_nested_json_from_image_batch(image_dataurls: List[str], batch_index: int) -> Dict[str, Any]:
    user_parts = [
        {
            "type": "text",
            "text": f"""These are scanned financial statement pages (batch {batch_index}).

Return JSON ONLY as a single JSON object.
- Use human-readable keys
- Nest where helpful (Income Statement / Balance Sheet / Cash Flow / Notes)
- Only include facts explicitly visible in the images (no guessing)
- Omit unclear values
"""
        }
    ]

    for img in image_dataurls:
        user_parts.append({"type": "image_url", "image_url": {"url": img}})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_parts},
    ]
    return aoai_chat_json(messages)