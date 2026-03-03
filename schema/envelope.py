ENVELOPE_SCHEMA = {
    "type": "object",
    "required": ["metadata", "schema", "payload", "evidence", "confidence", "validations"],
    "properties": {
        "metadata": {"type": "object"},
        "schema": {
            "type": "object",
            "required": ["schema_id", "schema_version"],
            "properties": {"schema_id": {"type": "string"}, "schema_version": {"type": "string"}},
        },
        "payload": {"type": "object"},
        "evidence": {"type": "object"},
        "confidence": {"type": "object"},
        "validations": {"type": "object"},
    },
    "additionalProperties": True,
}
