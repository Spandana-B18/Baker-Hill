

def build_table_snippet(ir: dict, max_tables: int = 6, max_cells: int = 180) -> str:
    parts = []
    tables = ir.get("tables", []) or []
    for ti, t in enumerate(tables[:max_tables], start=1):
        parts.append(f"TABLE {ti} rows={t.get('row_count')} cols={t.get('col_count')}")
        shown = 0
        for c in (t.get("cells") or []):
            if shown >= max_cells:
                parts.append("TABLE TRUNCATED")
                break
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            parts.append(f"r{c.get('row')} c{c.get('col')} {txt[:140]}")
            shown += 1
        parts.append("")
    return "\n".join(parts).strip()


def split_into_chunks(markdown: str, max_chars: int) -> list[str]:
    if not markdown:
        return []

    # Try page break marker split first (common in CU markdown)
    marker = "<!-- PageBreak -->"
    if marker in markdown:
        parts = markdown.split(marker)
        chunks = []
        current = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            add = p + "\n" + marker + "\n"
            if len(current) + len(add) > max_chars and current.strip():
                chunks.append(current)
                current = ""
            current += add
        if current.strip():
            chunks.append(current)
        return chunks

    # Fallback: fixed-size chunks
    return [markdown[i : i + max_chars] for i in range(0, len(markdown), max_chars) if markdown[i : i + max_chars].strip()]


def merge_envelopes(envelopes: list[dict]) -> dict:
    base = envelopes[0]

    for env in envelopes[1:]:
        # Prefer schema with higher selection confidence if present
        try:
            b_conf = float((base.get("confidence") or {}).get("schema_id_selection", 0))
            e_conf = float((env.get("confidence") or {}).get("schema_id_selection", 0))
            if e_conf > b_conf and env.get("schema"):
                base["schema"] = env["schema"]
                (base.setdefault("confidence", {}))["schema_id_selection"] = e_conf
        except Exception:
            pass

        # Payload merge
        bp = base.get("payload") or {}
        ep = env.get("payload") or {}
        if isinstance(bp, dict) and isinstance(ep, dict):
            for k, v in ep.items():
                if k not in bp or bp[k] in [None, "", [], {}]:
                    bp[k] = v
                else:
                    if isinstance(bp[k], list) and isinstance(v, list):
                        bp[k].extend(v)
        base["payload"] = bp

        # Evidence merge
        be = base.get("evidence") or {}
        ee = env.get("evidence") or {}
        if isinstance(be, dict) and isinstance(ee, dict):
            for k, v in ee.items():
                if k not in be:
                    be[k] = v
                else:
                    if isinstance(be[k], list) and isinstance(v, list):
                        be[k].extend(v)
        base["evidence"] = be

        # Confidence merge (keep max per key if numeric)
        bc = base.get("confidence") or {}
        ec = env.get("confidence") or {}
        if isinstance(bc, dict) and isinstance(ec, dict):
            for k, v in ec.items():
                try:
                    v_num = float(v)
                    b_num = float(bc.get(k, 0))
                    bc[k] = max(b_num, v_num)
                except Exception:
                    if k not in bc:
                        bc[k] = v
        base["confidence"] = bc

    base.setdefault("metadata", {})
    base["metadata"]["chunking"] = {"chunks": len(envelopes)}
    return base