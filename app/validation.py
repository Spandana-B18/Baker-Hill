from jsonschema import Draft202012Validator
from schema.envelope import ENVELOPE_SCHEMA
from schema.registry import SCHEMA_REGISTRY

def deterministic_validate(extracted: dict) -> dict:
    errors = []

    v_env = Draft202012Validator(ENVELOPE_SCHEMA)
    for e in v_env.iter_errors(extracted):
        errors.append({"type": "envelope_schema", "path": list(e.absolute_path), "message": e.message})

    schema_id = ((extracted.get("schema") or {}).get("schema_id") or "").strip()
    payload = extracted.get("payload") or {}

    if not schema_id:
        errors.append({"type": "schema_id", "path": ["schema", "schema_id"], "message": "schema_id is missing"})
    elif schema_id not in SCHEMA_REGISTRY:
        errors.append({"type": "schema_id", "path": ["schema", "schema_id"], "message": "schema_id not in registry"})
    else:
        v_payload = Draft202012Validator(SCHEMA_REGISTRY[schema_id])
        for e in v_payload.iter_errors(payload):
            errors.append({"type": "payload_schema", "path": ["payload"] + list(e.absolute_path), "message": e.message})

    return {"status": "pass" if not errors else "fail", "errors": errors}
