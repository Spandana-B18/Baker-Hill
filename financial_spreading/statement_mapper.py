"""
financial_spreading/statement_mapper.py

Maps extracted financial rows to the Chart of Accounts schema using
a precision first strategy:

1. Exact and keyword style match (fast path)
2. SAFE_MAP alias match (deterministic IRS label normalization)
3. Semantic similarity via Azure OpenAI embeddings (text-embedding-3-large)
4. Keeps all matched rows, including page 1 form fields and Schedule L rows
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from core.embeddings import AzureOpenAIEmbedder, EmbeddingError as _EmbeddingError
    _EMBEDDINGS_AVAILABLE = True
except Exception:
    _EMBEDDINGS_AVAILABLE = False


_STRIP_PUNCT = re.compile(r"[^a-z0-9 ]")
_STOPWORDS: Set[str] = {
    "and",
    "or",
    "the",
    "of",
    "in",
    "on",
    "to",
    "for",
    "from",
    "at",
    "by",
    "with",
    "current",
}

# ---------------------------------------------------------------------------
# Safe label normalization map
# ---------------------------------------------------------------------------
# Maps common tax-form label variants ? canonical terms that match the COA.
# Keys are lowercase. This is deterministic lookup -> NOT hallucination.
# Add entries here whenever a known tax-form label fails to match its COA line.
# ---------------------------------------------------------------------------
SAFE_MAP: Dict[str, str] = {
    # -- Form 1040 ------------------------------------------------------------
    "w-2":                         "wages",
    "w2":                          "wages",
    "w-2 income":                  "wages",
    "wages salaries tips":         "wages",
    "taxable interest":            "interest income",
    "tax-exempt interest":         "interest income",
    "ordinary dividends":          "dividends",
    "qualified dividends":         "dividends",
    "ira distributions":           "ira",
    "ira":                         "ira",
    "pensions and annuities":      "pension",
    "social security benefits":    "social security",
    "social security":             "social security",
    "adjusted gross income":       "agi",
    "total income":                "gross income",
    "gross income":                "gross income",
    "net profit or loss":          "net income",
    "net income or loss":          "net income",
    "total expenses":              "expenses",
    "total deductions":            "total operating expenses",
    "taxable income":              "taxable income",
    "federal income tax withheld": "taxes withheld",

    # -- Form 1065 / 1120-S / 1120 -> Income lines -----------------------------
    "gross receipts or sales":                          "sales",
    "gross receipts":                                   "sales",
    "returns and allowances":                           "sales returns and allowances",
    "cost of goods sold":                               "cost of goods sold",
    "gross profit":                                     "gross profit",
    "ordinary business income (loss)":                  "net income loss",
    "ordinary income (loss) from trade or business":    "net income loss",
    "net income (loss) per books":                      "net income",
    "guaranteed payments to partners":                  "officers salaries",
    "guaranteed payments for services":                 "officers salaries",
    "net earnings (loss) from self-employment":         "net income",

    # -- Form 1065 / 1120-S / 1120 -> Deduction lines --------------------------
    "compensation of officers":                                   "officers salaries",
    "salaries and wages":                                         "wages",
    "salaries and wages (less employment credits)":               "wages",
    "salaries and wages (other than to partners)":                "wages",
    "salaries and wages (other than to partners) (less employment credits)": "wages",
    "repairs and maintenance":                                    "other operating expenses",
    "bad debts":                                                  "bad debt expense",
    "rent":                                                       "lease rent expense",
    "rents":                                                      "lease rent expense",
    "taxes and licenses":                                         "other operating expenses",
    "advertising":                                                "other operating expenses",
    "depreciation":                                               "depreciation",
    "depreciation (see instructions)":                            "depreciation",
    "depletion":                                                  "depreciation",
    "retirement plans etc":                                       "other operating expenses",
    "employee benefit programs":                                  "other operating expenses",
    "other deductions":                                           "other operating expenses",
    "interest":                                                   "interest expense",
    "interest (see instructions)":                                "interest expense",
    "interest expense":                                           "interest expense",
    "total deductions":                                           "total operating expenses",
    "taxable income before net operating loss deduction":         "profit before tax",
    "taxable income before nol deduction":                        "profit before tax",
    "income tax":                                                 "current taxes",
    "total tax":                                                  "current taxes",

    # -- Schedule L -> Balance Sheet --------------------------------------------
    "cash":                                                       "cash",
    "trade notes and accounts receivable":                        "trade accounts receivable",
    "trade notes and accounts receivable (less allowance)":       "trade accounts receivable",
    "less allowance for bad debts":                               "reserve for bad debts",
    "inventories":                                                "total inventory",
    "us government obligations":                                  "marketable securities",
    "tax-exempt securities":                                      "marketable securities",
    "other current assets":                                       "other current assets",
    "loans to partners":                                          "due from affiliates",
    "loans to partners (or persons related to partners)":         "due from affiliates",
    "loans to shareholders":                                      "due from stockholders",
    "buildings and other depreciable assets":                     "property plant equipment",
    "less accumulated depreciation":                              "accumulated depreciation",
    "depletable assets":                                          "property plant equipment",
    "less accumulated depletion":                                 "accumulated depreciation",
    "land (net of any amortization)":                             "land",
    "intangible assets (amortizable only)":                       "intangible assets",
    "less accumulated amortization":                              "accumulated depreciation",
    "other assets":                                               "other long term assets",
    "total assets (see instructions)":                            "total assets",
    "total assets":                                               "total assets",
    "accounts payable":                                           "accounts payable",
    "mortgages notes bonds payable in less than 1 year":          "notes payable banks",
    "mortgages, notes, bonds payable in less than 1 year":        "notes payable banks",
    "other current liabilities":                                  "other current liabilities",
    "all nonrecourse loans":                                      "notes payable banks",
    "mortgages notes bonds payable in 1 year or more":            "long term debt",
    "mortgages, notes, bonds payable in 1 year or more":          "long term debt",
    "other liabilities":                                          "other long term liabilities",
    "partners capital accounts":                                  "total net worth",
    "partners' capital accounts":                                 "total net worth",
    "loans from partners":                                        "due to related",
    "loans from partners (or persons related to partners)":       "due to related",
    "loans from shareholders":                                    "due to stockholders",
    "capital stock":                                              "common stock",
    "additional paid-in capital":                                 "additional paid in capital",
    "retained earnings":                                          "retained earnings",
    "total liabilities and capital":                              "total liabilities and equity",
    "total liabilities and stockholders equity":                  "total liabilities and equity",
    "total liabilities and partners capital":                     "total liabilities and equity",
}


def _apply_safe_map(label: str) -> str:
    """
    Return the canonical alias for a label if one exists in SAFE_MAP.
    Falls back to returning the label unchanged if no alias is found.

    Matching order:
      1. Exact lowercase match
      2. Prefix match -> handles labels like
         "Ordinary business income (loss) (page 1, line 22)" which
         start with a known SAFE_MAP key but have extra form metadata.

    Example:
        "W-2 income"  ?  "wages"
        "Taxable interest"  ?  "interest income"
        "Total assets"  ?  "Total assets"   (no alias, returned as-is)
    """
    normalized = label.strip().lower()
    if normalized in SAFE_MAP:
        return SAFE_MAP[normalized]
    # Prefix match: key must be followed by a space, '(' or end-of-string
    for key, value in SAFE_MAP.items():
        if normalized.startswith(key) and (
            len(normalized) == len(key)
            or normalized[len(key)] in (" ", "(", ",", ";")
        ):
            return value
    return label


def _normalize(text: str) -> str:
    """
    Lower case, strip punctuation, collapse whitespace.
    """
    text = text or ""
    text = text.lower().strip()
    text = _STRIP_PUNCT.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Matches form cross-reference metadata appended to labels, e.g.:
#   "(page 1, line 22)"  "(line 22)"  "(see instructions)"  "(attach statement)"
_FORM_META = re.compile(
    r"\("
    r"(?:page\s+\d+[,\s]*)?"
    r"(?:line[s]?\s+[\d,\s]+)?"
    r"(?:see\s+instructions)?"
    r"(?:attach\s+[^)]+)?"
    r"\)",
    re.IGNORECASE,
)


def _canonicalize_source_label(text: str) -> str:
    """
    Normalize tax form style labels before matching.

    Strips form cross-reference metadata so that labels like:
      "Ordinary business income (loss) (page 1, line 22)"
    become:
      "Ordinary business income (loss)"

    Also truncates IRS instruction sentences appended to label text:
      "Gross income. Subtract line 49 from line 44. Enter the result here..."
    becomes:
      "Gross income"

      "Wages, salaries, tips, etc. Attach Form(s) W-2"
    becomes:
      "Wages, salaries, tips, etc"

    Other examples:
      "F Total assets (see instructions)"   -> "Total assets"
      "Total assets [Beginning of tax year]" -> "Total assets beginning of tax year"
    """
    t = text or ""
    # Strip OCR form-control markers emitted by Azure Content Understanding
    # for checkbox fields: ":selected:" / ":unselected:"
    t = re.sub(r":\s*(?:selected|unselected)\s*:", " ", t, flags=re.IGNORECASE)
    # Block accounting-method checkbox labels (Schedule C/F: "Cash Accrual").
    # The word "accrual" never appears in a genuine financial line label.
    if re.search(r"\baccrual\b", t, re.IGNORECASE):
        return ""
    t = re.sub(r"^[A-Z]\s+", "", t)
    t = _FORM_META.sub("", t)               # strip (page X, line Y) and similar
    # Truncate at first sentence boundary: ". Uppercase+lowercase" marks start
    # of an IRS instruction sentence appended after the actual financial label.
    t = re.sub(r"\.\s+[A-Z][a-z].*$", "", t)
    t = re.sub(r"[\[\]]", " ", t)           # brackets -> space
    t = re.sub(r"[\.\:]+", " ", t)          # dots/colons -> space
    t = t.replace("&", " and ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> Set[str]:
    norm = _normalize(text)
    if not norm:
        return set()
    return {tok for tok in norm.split() if tok and tok not in _STOPWORDS}


def _is_substantive(norm_label: str, min_tokens: int = 2) -> bool:
    """
    Return True if the normalized label has at least `min_tokens` meaningful tokens
    (length = 3, not a stopword).

    Used to guard the substring-containment direction where the document label
    is shorter than the COA keyword -> e.g. "income" should NOT match
    "Federal Income Tax Receivable" even though "income" ? that string.
    """
    tokens = [t for t in norm_label.split() if len(t) >= 3 and t not in _STOPWORDS]
    return len(tokens) >= min_tokens


# ---------------------------------------------------------------------------
# Semantic similarity via Azure OpenAI embeddings
# ---------------------------------------------------------------------------

# Module-level embedder singleton (created lazily)
_embedder: Optional[Any] = None
_embedder_init_failed: bool = False

# COA embedding index cache: id(schema) -> {"items": [...], "vectors": [...]}
_COA_INDEX_CACHE: Dict[int, Dict[str, Any]] = {}


def _get_embedder() -> Optional[Any]:
    global _embedder, _embedder_init_failed
    if _embedder_init_failed or not _EMBEDDINGS_AVAILABLE:
        return None
    if _embedder is None:
        try:
            _embedder = AzureOpenAIEmbedder()
        except Exception:
            _embedder_init_failed = True
            return None
    return _embedder


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _get_coa_index(schema: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Build (and cache) one embedding per COA entry.
    Text = row_label + all keywords joined -> gives the model full semantic context.
    """
    key = id(schema)
    if key in _COA_INDEX_CACHE:
        return _COA_INDEX_CACHE[key]

    embedder = _get_embedder()
    if embedder is None:
        return None

    texts = []
    for item in schema:
        parts = [item.get("row_label", "")]
        parts.extend(item.get("keywords", []))
        texts.append(" | ".join(p for p in parts if p))

    try:
        vectors = embedder.embed_texts(texts)
    except Exception:
        return None

    _COA_INDEX_CACHE[key] = {"items": schema, "vectors": vectors}
    return _COA_INDEX_CACHE[key]


def _semantic_match_single(
    label: str,
    schema: List[Dict[str, Any]],
    coa_index: Dict[str, Any],
    threshold: float = 0.75,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Embed `label` and return the best (coa_item, cosine_score) >= threshold.
    Returns (None, 0.0) if no match or embedding fails.
    """
    embedder = _get_embedder()
    if embedder is None:
        return None, 0.0
    try:
        vecs = embedder.embed_texts([label])
    except Exception:
        return None, 0.0

    query_vec = vecs[0]
    best_item = None
    best_score = 0.0
    for item, coa_vec in zip(coa_index["items"], coa_index["vectors"]):
        score = _cosine(query_vec, coa_vec)
        if score > best_score:
            best_score = score
            best_item = item
    if best_item and best_score >= threshold:
        return best_item, best_score
    return None, 0.0


def _keyword_exact_match(
    label: str,
    schema: List[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], float, str]]:
    """
    Check whether normalized label contains any schema keyword or vice versa.
    Returns best (item, score, matched_candidate) or None.

    Scoring:
      0.97 -> exact match against the COA row_label itself
      0.95 -> exact match against a keyword alias
      0.90 -> substring containment (either direction)
    Row-label matches score higher than keyword matches to break ties correctly
    (e.g. "Gross profit" should beat "total income" for L0227 Gross Profit).

    Empty labels are rejected immediately -> an empty string is a substring of
    every string in Python, which would otherwise cause every COA line to match.
    """
    norm_label = _normalize(label)
    if not norm_label:          # guard: empty after strip ? never match anything
        return None

    best_item: Optional[Dict[str, Any]] = None
    best_score = 0.0
    best_candidate = ""

    for item in schema:
        row_label = item["row_label"]
        keywords  = item.get("keywords", [])
        # row_label first (index 0), then keywords -> so we can distinguish them
        for idx, candidate in enumerate([row_label] + keywords):
            norm_cand = _normalize(candidate)
            if not norm_cand:
                continue

            if norm_cand == norm_label:
                # Exact match: row_label wins over keyword alias
                score = 0.97 if idx == 0 else 0.95
            elif norm_cand in norm_label:
                # COA keyword found inside document label -> valid containment
                score = 0.90
            elif norm_label in norm_cand and _is_substantive(norm_label):
                # Document label inside COA keyword -> only valid when the label
                # has = 2 meaningful tokens; prevents "income", "Son", "taxes"
                # etc. from matching any COA entry that contains that word.
                score = 0.90
            else:
                continue

            if score > best_score:
                best_score = score
                best_item = item
                best_candidate = candidate

    if best_item is None:
        return None

    return best_item, best_score, best_candidate


def match_row(
    row_label: str,
    schema: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """
    Return (best_schema_item, score, matched_candidate) for row_label.

    Strategy (in order):
      1. Keyword exact/substring match on cleaned label         (score 0.90-0.97)
      2. Keyword exact/substring match on SAFE_MAP alias        (score 0.90-0.97)
      3. Semantic similarity via Azure OpenAI embeddings        (score 0.0-1.0)
    """
    cleaned_label = _canonicalize_source_label(row_label)
    mapped_label  = _apply_safe_map(cleaned_label)

    # Step 1 -> exact/keyword match on cleaned label
    exact = _keyword_exact_match(cleaned_label, schema)
    if exact:
        return exact

    # Step 2 -> exact/keyword match on SAFE_MAP alias (if alias differs)
    if mapped_label != cleaned_label:
        exact = _keyword_exact_match(mapped_label, schema)
        if exact:
            return exact

    # Step 3 -> semantic similarity via Azure OpenAI embeddings
    coa_index = _get_coa_index(schema)
    if coa_index is None:
        return None, 0.0, ""

    # Try both the cleaned label and the SAFE_MAP alias; keep best
    best_item: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for try_label in {cleaned_label, mapped_label}:
        if not _normalize(try_label):
            continue
        item, score = _semantic_match_single(try_label, schema, coa_index)
        if score > best_score:
            best_score = score
            best_item = item

    matched_candidate = best_item["row_label"] if best_item else ""
    return best_item, best_score, matched_candidate


def _build_reasoning(
    source_label: str,
    matched_candidate: str,
    match: Dict[str, Any],
    score: float,
    page: Optional[int],
    year: Optional[int],
) -> str:
    page_str = f" on page {page}" if page else ""
    year_str = f" for year {year}" if year else ""

    if score >= 0.90:
        method = "exact keyword match"
    elif score >= 0.75:
        method = "high confidence semantic match"
    else:
        method = "semantic similarity match"

    return (
        f"Matched '{source_label}'{page_str}{year_str} to "
        f"'{match['row_label']}' ({match['chart_of_account_line']}) "
        f"using candidate '{matched_candidate}' via {method} "
        f"(score: {score:.2f})."
    )


def _build_reference(page: Optional[int], period: Optional[str]) -> str:
    parts = [f"Page {page}" if page is not None and page > 0 else ""]
    if period:
        parts.append(period)
    return ", ".join(p for p in parts if p)


def spread_statement(
    rows: List[Dict[str, Any]],
    schema: List[Dict[str, Any]],
    threshold: float = 0.75,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Map a list of extracted rows to the COA schema.

    Output row contract:
        chart_of_account_line, row_label, year, value,
        confidence, reference, source_label, reasoning
    """
    mapped_rows: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []

    for row in rows:
        source_label = row.get("label", "")
        # Skip noise labels that canonicalize to empty (e.g. "(see instructions)")
        if not _normalize(_canonicalize_source_label(source_label)):
            continue
        match, score, matched_candidate = match_row(source_label, schema)

        if match is None or score < threshold:
            unmatched.append(
                {
                    "source_label": source_label,
                    "values": row.get("values", []),
                    "page": row.get("page"),
                    "match_score": round(score, 3),
                }
            )
            continue

        values_list = row.get("values", [])
        has_period_values = any(
            (v.get("period") if isinstance(v, dict) else None) is not None
            for v in values_list
        )
        for val_entry in values_list:
            value          = val_entry["value"] if isinstance(val_entry, dict) else val_entry
            original_value = val_entry.get("original_value", "") if isinstance(val_entry, dict) else ""
            year           = val_entry.get("year") if isinstance(val_entry, dict) else None
            period         = val_entry.get("period") if isinstance(val_entry, dict) else None
            if period is None and has_period_values:
                continue
            if year is not None:
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    pass

            reasoning = _build_reasoning(
                source_label=source_label,
                matched_candidate=matched_candidate,
                match=match,
                score=score,
                page=row.get("page"),
                year=year,
            )

            output_row_label = match["row_label"]
            if period:
                output_row_label = f"{output_row_label} [{period}]"

            mapped_rows.append(
                {
                    "chart_of_account_line": match["chart_of_account_line"],
                    "row_label":             output_row_label,
                    "year":                  year,
                    "value":                 value,
                    "original_value":        original_value,
                    "confidence":            round(score, 3),
                    "reference":             _build_reference(row.get("page"), period),
                    "source_label":          source_label,
                    "reasoning":             reasoning,
                }
            )

    # Post-process: for each COA line, if period-qualified rows exist,
    # remove bare (no-period) rows to avoid title duplication.
    from collections import defaultdict as _dd
    coa_has_period = _dd(bool)
    for r in mapped_rows:
        rl = r.get('row_label', '')
        if '[Beginning of tax year]' in rl or '[End of tax year]' in rl:
            coa_has_period[r['chart_of_account_line']] = True
    mapped_rows = [
        r for r in mapped_rows
        if not (
            coa_has_period[r['chart_of_account_line']]
            and '[Beginning of tax year]' not in r.get('row_label', '')
            and '[End of tax year]' not in r.get('row_label', '')
        )
    ]

    return mapped_rows, unmatched


def map_coa_to_document(
    candidates: List[Dict[str, Any]],
    schema: List[Dict[str, Any]],
    threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    COA-driven mapping: loop every COA entry and find the best matching
    candidate from the document.

    This is the INVERSE of spread_statement():
      - spread_statement()     loops document rows ? tries to find a COA match
      - map_coa_to_document()  loops COA entries   ? searches document candidates

    Rules:
      - Only emits a row when a candidate scores >= threshold (default 0.75)
      - Uses SAFE_MAP aliases so tax-form labels like "W-2" match "Wages" in COA
      - Never creates a value that isn't in the document -> no hallucination possible
      - If no candidate meets the threshold for a COA line, that line is simply skipped
      - Emits one row per period (Beginning/End of tax year) for Schedule L tables

    Parameters
    ----------
    candidates : flat list from extract_all_candidates(cu_json)
                 each item: {label, value, original_value, page, year, period}
    schema     : loaded COA schema -> list of dicts with row_label,
                 chart_of_account_line, keywords
    threshold  : minimum score to accept a match (default 0.75)

    Returns
    -------
    List of mapped row dicts using the same contract as spread_statement()
    """
    mapped_rows: List[Dict[str, Any]] = []

    # -- Pre-compute semantic embeddings for all unique candidate labels ------
    # One batch API call covers every label in the document.
    # Falls back gracefully to keyword-only matching if embeddings unavailable.
    _label_vectors: Dict[str, List[float]] = {}
    _coa_index = _get_coa_index(schema)
    # Map each COA item id ? its vector index for O(1) lookup in the inner loop
    _coa_item_idx: Dict[int, int] = {}
    if _coa_index is not None:
        _coa_item_idx = {id(it): i for i, it in enumerate(_coa_index["items"])}

        unique_labels: List[str] = []
        seen: Set[str] = set()
        for _c in candidates:
            for _lbl in {
                _canonicalize_source_label(_c.get("label", "")),
                _apply_safe_map(_canonicalize_source_label(_c.get("label", ""))),
            }:
                if _lbl and _normalize(_lbl) and _lbl not in seen:
                    unique_labels.append(_lbl)
                    seen.add(_lbl)
        if unique_labels:
            _emb = _get_embedder()
            if _emb is not None:
                try:
                    _vecs = _emb.embed_texts(unique_labels)
                    _label_vectors = dict(zip(unique_labels, _vecs))
                except Exception:
                    pass

    for coa_item in schema:
        coa_label    = coa_item.get("row_label", "")
        coa_code     = coa_item.get("chart_of_account_line", "")
        coa_keywords = coa_item.get("keywords", [])
        all_coa_targets = [coa_label] + coa_keywords

        # Track best-scoring candidate per period so that multi-period tables
        # (Schedule L: Beginning / End of tax year) emit one row per period
        # instead of dropping all but the single best-scoring candidate.
        # Key: period value (str or None); Value: (candidate, score)
        best_per_period: Dict[Optional[str], Tuple[Optional[Dict[str, Any]], float]] = {}

        for candidate in candidates:
            raw_label        = candidate.get("label", "")
            clean_label      = _canonicalize_source_label(raw_label)
            normalized_label = _apply_safe_map(clean_label)

            # Skip candidates whose label is entirely noise after canonicalization.
            # "(see instructions)", "(page 1, line 22)" etc. strip to "" and would
            # otherwise match every COA entry via the empty-string-in-any-string rule.
            if not _normalize(clean_label) and not _normalize(normalized_label):
                continue

            val     = candidate.get("value") or 0
            abs_val = abs(val)
            is_line_ref = (
                0 < abs_val < 100
                and float(abs_val) == int(float(abs_val))
            )
            line_ref_penalty = 0.55 if is_line_ref else 1.0

            period = candidate.get("period")

            # -- Step 1: keyword exact/substring match --------------------
            exact_score = 0.0
            for try_label in {clean_label, normalized_label}:
                norm = _normalize(try_label)
                if not norm:          # guard: empty string matches everything
                    continue
                for t_idx, coa_target in enumerate(all_coa_targets):
                    norm_coa = _normalize(coa_target)
                    if not norm_coa:
                        continue
                    if norm_coa == norm:
                        # row_label (idx 0) wins ties over keyword aliases
                        s = 0.97 if t_idx == 0 else 0.95
                    elif norm_coa in norm:
                        # COA keyword found inside document label -> valid
                        s = 0.90
                    elif norm in norm_coa and _is_substantive(norm):
                        # Document label inside COA keyword -> only when label
                        # has = 2 meaningful tokens (blocks "income", "Son", etc.)
                        s = 0.90
                    else:
                        continue
                    if s > exact_score:
                        exact_score = s

            if exact_score > 0:
                score = exact_score * line_ref_penalty
                prev_score = best_per_period.get(period, (None, 0.0))[1]
                if score > prev_score:
                    best_per_period[period] = (candidate, score)
                continue  # exact match wins; skip semantic

            # -- Step 2: semantic similarity via Azure OpenAI embeddings ------
            if _coa_index is not None and _label_vectors:
                coa_idx = _coa_item_idx.get(id(coa_item))
                if coa_idx is not None:
                    coa_vec = _coa_index["vectors"][coa_idx]
                    for try_label in {clean_label, normalized_label}:
                        query_vec = _label_vectors.get(try_label)
                        if query_vec is None:
                            continue
                        raw_score = _cosine(query_vec, coa_vec)
                        score = raw_score * line_ref_penalty

                        prev_score = best_per_period.get(period, (None, 0.0))[1]
                        if score > prev_score:
                            best_per_period[period] = (candidate, score)

        # Emit one mapped row per period that meets the threshold
        for period, (best_candidate, best_score) in best_per_period.items():
            if best_candidate is None or best_score < threshold:
                continue

            output_row_label = coa_label
            if period:
                output_row_label = f"{output_row_label} [{period}]"

            reasoning = _build_reasoning(
                source_label=best_candidate.get("label", ""),
                matched_candidate=coa_label,
                match=coa_item,
                score=best_score,
                page=best_candidate.get("page"),
                year=best_candidate.get("year"),
            )

            raw_year = best_candidate.get("year")
            if raw_year is not None:
                try:
                    raw_year = int(raw_year)
                except (ValueError, TypeError):
                    pass

            mapped_rows.append({
                "chart_of_account_line": coa_code,
                "row_label":             output_row_label,
                "year":                  raw_year,
                "value":                 best_candidate.get("value"),
                "original_value":        best_candidate.get("original_value", ""),
                "confidence":            round(best_score, 3),
                "reference":             _build_reference(best_candidate.get("page"), period),
                "source_label":          best_candidate.get("label", ""),
                "reasoning":             reasoning,
            })

    # Post-process: if a COA line has period-qualified rows, drop bare rows.
    from collections import defaultdict as _dd2
    coa_has_period2 = _dd2(bool)
    for r in mapped_rows:
        rl = r.get("row_label", "")
        if "[Beginning of tax year]" in rl or "[End of tax year]" in rl:
            coa_has_period2[r["chart_of_account_line"]] = True
    mapped_rows = [
        r for r in mapped_rows
        if not (
            coa_has_period2[r["chart_of_account_line"]]
            and "[Beginning of tax year]" not in r.get("row_label", "")
            and "[End of tax year]" not in r.get("row_label", "")
        )
    ]

    return mapped_rows
