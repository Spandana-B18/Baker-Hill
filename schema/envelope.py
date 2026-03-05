ENVELOPE_SCHEMA = {
    "type": "object", # the envelope is a JSON object
    "required": ["metadata", "schema", "payload", "evidence", "confidence", "validations"], # the envelope is required to have these properties
    "properties": {
        "metadata": {"type": "object"}, # metadata about the document
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
