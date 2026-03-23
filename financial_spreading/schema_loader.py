"""
financial_spreading/schema_loader.py

Loads the correct Chart-of-Accounts schema JSON file.

Supported schemas:
- Data/coa_schema.json              -> default financial / generic COA
- Data/tax_coa_schema.json          -> tax COA schema

The caller can:
1. pass an explicit path
2. pass a document_type
3. rely on default fallback
"""

import json
import os
from typing import Any, Dict, List, Optional


def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def _data_path(file_name: str) -> str:
    return os.path.join(_project_root(), "Data", file_name)


def resolve_coa_schema_path(
    *,
    document_type: Optional[str] = None,
    schema_name: Optional[str] = None,
) -> str:
    """
    Resolve the schema file path.

    Priority:
    1. explicit schema_name if provided
    2. document_type-based routing
    3. fallback to coa_schema.json
    """
    if schema_name:
        return _data_path(schema_name)

    doc_type = str(document_type or "").strip().lower()

    if doc_type == "tax_document":
        return _data_path("tax_coa_schema.json")

    if doc_type == "financial_document":
        return _data_path("coa_schema.json")

    return _data_path("coa_schema.json")


def load_coa_schema(
    path: Optional[str] = None,
    *,
    document_type: Optional[str] = None,
    schema_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load a COA schema.

    Parameters
    ----------
    path : explicit full file path
    document_type : e.g. tax_document / financial_document
    schema_name : explicit file name inside Data/, e.g. tax_coa_schema.json

    Returns
    -------
    list[dict]
    """
    resolved_path = path or resolve_coa_schema_path(
        document_type=document_type,
        schema_name=schema_name,
    )

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"COA schema not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)

    if not isinstance(schema, list):
        raise ValueError(f"{os.path.basename(resolved_path)} must be a JSON array of objects.")

    required_keys = {"chart_of_account_line", "row_label"}
    for i, item in enumerate(schema):
        if not isinstance(item, dict):
            raise ValueError(f"Schema row at index {i} must be a JSON object.")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(
                f"Schema row at index {i} is missing required keys: {sorted(missing)}"
            )

    return schema