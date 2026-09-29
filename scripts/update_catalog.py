"""Regenerate src/tarjuman/data/models.json from models.dev.

    uv run python scripts/update_catalog.py            # fetch https://models.dev/api.json
    uv run python scripts/update_catalog.py file.json  # or use a saved copy

Keeps only the providers that data/providers.json points at (its "catalog" keys), and only
the fields Tarjuman uses. Corrections go in data/overrides.json, never in the output."""

from __future__ import annotations

import datetime
import json
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "src" / "tarjuman" / "data"
SOURCE = "https://models.dev/api.json"
LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")


def convert(m: dict) -> dict:
    out: dict = {"name": m.get("name") or m["id"]}
    limit = m.get("limit") or {}
    if limit.get("context"):
        out["context"] = limit["context"]
    if limit.get("output"):
        out["max_output"] = limit["output"]
    out["vision"] = "image" in ((m.get("modalities") or {}).get("input") or [])
    if not m.get("tool_call"):
        out["tools"] = False
    if m.get("temperature") is False:
        out["temperature"] = False
    if m.get("reasoning"):
        out["reasoning"] = True
        opts = {o["type"]: o for o in m.get("reasoning_options") or []}
        if "effort" in opts:
            values = [v for v in opts["effort"].get("values") or [] if v]
            levels = [lv for lv in LEVELS if lv in values]
            if "none" in values or "toggle" in opts:
                levels.insert(0, "off")
            out["thinking"], out["levels"] = "effort", levels
        elif "budget_tokens" in opts:
            b = opts["budget_tokens"]
            out["thinking"] = "budget"
            if b.get("min") or b.get("max"):
                out["budget"] = [b.get("min") or 1024, b.get("max")]  # max None: unknown
        elif "toggle" in opts:
            out["thinking"] = "toggle"
        field = (m.get("interleaved") or {}).get("field")
        if field:
            out["reasoning_field"] = field
    c = m.get("cost") or {}
    if any(c.get(k) for k in ("input", "output")):
        out["price"] = {k: c.get(k) or 0 for k in ("input", "output", "cache_read", "cache_write")}
    if m.get("status"):
        out["status"] = m["status"]
    return out


def main() -> None:
    if len(sys.argv) > 1:
        raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        req = urllib.request.Request(SOURCE, headers={"User-Agent": "tarjuman-catalog"})
        raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    source = json.loads(raw)
    wanted = sorted({p.get("catalog", name) for name, p in
                     json.loads((DATA / "providers.json").read_text("utf-8")).items()
                     if not name.startswith("_") and p.get("catalog", name)})
    providers = {}
    for name in wanted:
        models = (source.get(name) or {}).get("models") or {}
        providers[name] = {mid: convert(m) for mid, m in sorted(models.items())}
    out = {"source": SOURCE, "generated": datetime.date.today().isoformat(), "providers": providers}
    (DATA / "models.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    print(", ".join(f"{p}: {len(m)}" for p, m in providers.items()))


if __name__ == "__main__":
    main()
