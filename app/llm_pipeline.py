import sys
from pathlib import Path

# Ensure project root is on path so services, utils, schema, config resolve
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from services.azure_openai import azure_openai_client
from utils.ir_processing import build_table_snippet, split_into_chunks, merge_envelopes
from schema.registry import SCHEMA_REGISTRY
from schema.envelope import ENVELOPE_SCHEMA
import json
from config import AZURE_OPENAI_DEPLOYMENT

def llm_dynamic_json(ir: dict, user_hint: str = "", chunk_chars: int = 18000) -> dict:
    """
    Flow: (1) build system_msg + split markdown into chunks
          (2) for each chunk: build user_msg (schema, envelope shape, chunk, tables)
          (3) call Azure OpenAI with messages=[system_msg, user_msg]  <-- LLM "listens" to system_msg here
          (4) parse JSON reply and merge all chunk results
    """
    client = azure_openai_client()

    markdown_full = (ir.get("markdown") or "").strip()
    chunks = split_into_chunks(markdown_full, max_chars=chunk_chars)
    if not chunks:
        raise RuntimeError("No markdown content to process")

    schema_ids = "\n".join(sorted(SCHEMA_REGISTRY.keys()))
    table_snip = build_table_snippet(ir)

    system_msg = """
You are a deterministic financial document parser.

You will receive OCR or markdown extracted from a financial document.
The layout, tables, spacing, and sections may vary.

Follow these rules strictly:

---------------------------------
STRUCTURE DETECTION
---------------------------------

1. Identify section titles (e.g., Assets, Liabilities).
2. Detect tables ONLY if:
   - A header row exists
   - There are 2 or more consistent columns

---------------------------------
TABLE EXTRACTION RULES (CRITICAL)
---------------------------------

1. First detect the header row.
2. Count the number of columns in the header.
3. Every row under that header must map values strictly by position.
4. The first column is always the row label (key).
5. Remaining columns must map EXACTLY to header columns in order.
6. If a row has fewer values than headers → pad with null.
7. If a row has extra values → store extras under:
   "unmapped_values": []
8. Never shift numbers left or right.
9. Never merge two numeric columns.
10. Preserve numbers exactly as written.

---------------------------------
KEY-VALUE (NON-TABLE) RULES
---------------------------------

If content is not part of a table:
- Extract as direct key-value pair.
- If multiple numbers appear on same line and no headers exist,
  store as array of values instead of guessing.

---------------------------------
AMBIGUITY RULE
---------------------------------

If alignment is unclear:
- Do NOT guess.
- Store the full line as:
  {
    "raw_row": "original text"
  }

---------------------------------
OUTPUT RULES
---------------------------------

- Output strictly valid JSON.
- Preserve hierarchy.
- Do NOT infer missing fields.
- Do NOT normalize field names.
- Do NOT summarize.
- Do NOT hallucinate.
- Return JSON only.

"""

    envelopes = []
    for idx, chunk in enumerate(chunks, start=1):
        user_msg = (
            f"Allowed schema_id values:\n{schema_ids}\n\n"
            "Envelope shape:\n"
            "{\n"
            '  "metadata": { },\n'
            '  "schema": { "schema_id": "", "schema_version": "" },\n'
            '  "payload": { },\n'
            '  "evidence": { },\n'
            '  "confidence": { },\n'
            '  "validations": { }\n'
            "}\n\n"
            "Payload must be dynamically generated from the document: add whatever keys and values you find "
            "(headings, table columns, form fields, numbers, dates). Use snake_case for keys.\n\n"
            f"User hint: {user_hint}\n"
            f"Chunk {idx} of {len(chunks)}\n\n"
            "Document markdown chunk:\n"
            f"{chunk}\n\n"
            "Global tables snippet:\n"
            f"{table_snip}\n"
        )

        # --- LLM receives system_msg here (once per chunk) ---
        # messages[0] = system: sets model behavior (parser instructions).
        # messages[1] = user: chunk + schema + tables. Model uses both to produce JSON.
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.2,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"},
        )
        envelopes.append(json.loads(resp.choices[0].message.content))

    return merge_envelopes(envelopes)


def llm_business_validation(ir: dict, extracted: dict, chunk_chars: int = 12000) -> dict:
    client = azure_openai_client()

    markdown_full = (ir.get("markdown") or "").strip()
    chunks = split_into_chunks(markdown_full, max_chars=chunk_chars)
    if not chunks:
        return {"issues": [], "overall_risk": "low"}

    system_msg = (
        "You validate extracted business data.\n"
        "Return JSON only.\n"
        "Find inconsistencies, missing fields, suspicious values.\n"
        "Tie each issue to evidence from the document chunk.\n"
    )

    all_issues = []
    overall = "low"

    for idx, chunk in enumerate(chunks, start=1):
        user_msg = (
            f"Chunk {idx} of {len(chunks)}\n\n"
            "Document markdown chunk:\n"
            f"{chunk}\n\n"
            "Extracted JSON:\n"
            f"{json.dumps(extracted, ensure_ascii=False)}\n\n"
            "Return JSON with keys:\n"
            "issues: list of {path, severity, description, suggested_action, evidence}\n"
            "overall_risk: low or medium or high\n"
        )

        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            temperature=0.1,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)

        all_issues.extend(out.get("issues", []) or [])
        r = (out.get("overall_risk") or "low").lower()
        if r == "high":
            overall = "high"
        elif r == "medium" and overall != "high":
            overall = "medium"

    return {"issues": all_issues, "overall_risk": overall}