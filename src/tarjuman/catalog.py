"""What we know about each model: limits, vision, how it reasons, and what it costs.

Generated from models.dev (scripts/update_catalog.py -> data/models.json), corrected by
data/overrides.json. All of it is optional knowledge: an unknown model still works, with
safe defaults (vision assumed, the provider's own reasoning default, cost None)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

from .types import Message, Usage

# Every level any provider exposes, weakest to strongest. Requests use the neutral subset
# (types.ReasoningLevel); "xhigh" only appears in what models accept.
LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

# Anthropic prices a 1 h cache write at 2x the base input price (a 5 min write is 1.25x).
LONG_WRITE_TIMES_INPUT = 2.0


@dataclass(frozen=True)
class Price:
    """USD per million tokens (the base tier: long-context surcharges are not modelled)."""
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    id: str
    name: str = ""
    context: int | None = None
    max_output: int | None = None
    vision: bool | None = None
    tools: bool = True
    temperature: bool = True               # False: the model rejects a temperature
    reasoning: bool = False
    thinking: str | None = None            # how reasoning is set: "effort" | "budget" | "toggle"
    levels: tuple[str, ...] | None = None  # effort levels accepted ("off" if it can stop)
    budget: tuple[int, int | None] | None = None  # budget_tokens min, max (None: unknown)
    reasoning_field: str | None = None     # its own reasoning must come back in this field
    price: Price | None = None
    status: str | None = None              # "deprecated", ...


def _read(name: str) -> dict[str, Any]:
    try:
        return json.loads(resources.files("tarjuman.data").joinpath(name).read_text("utf-8"))
    except FileNotFoundError:
        return {}


@cache
def _table() -> dict[str, dict[str, dict[str, Any]]]:
    models = _read("models.json").get("providers", {})
    for provider, entries in _read("overrides.json").get("providers", {}).items():
        for model_id, fields in entries.items():
            models.setdefault(provider, {}).setdefault(model_id, {}).update(fields)
    return models


def lookup(provider: str, model: str) -> ModelInfo | None:
    d = _table().get(provider, {}).get(model)
    if d is None:
        return None
    return ModelInfo(
        provider, model, d.get("name", ""), d.get("context"), d.get("max_output"),
        d.get("vision"), d.get("tools", True), d.get("temperature", True),
        d.get("reasoning", False), d.get("thinking"),
        tuple(d["levels"]) if d.get("levels") else None,
        tuple(d["budget"]) if d.get("budget") else None,
        d.get("reasoning_field"), Price(**d["price"]) if d.get("price") else None,
        d.get("status"))


def nearest(level: str, supported: tuple[str, ...] | None) -> str:
    """The supported level closest to `level`; ties go to the weaker one. Unknown: unchanged."""
    if not supported or level in supported:
        return level
    rank = {lv: i for i, lv in enumerate(LEVELS)}
    options = [lv for lv in supported if lv in rank]
    if not options or level not in rank:
        return level
    return min(options, key=lambda lv: (abs(rank[lv] - rank[level]), rank[lv]))


def cost(usage: Usage, info: ModelInfo | None) -> float | None:
    """USD for one response, or None when the price is unknown (never a guess)."""
    if info is None or info.price is None:
        return None
    p = info.price
    short_writes = usage.cache_write - usage.cache_write_long
    return (usage.input * p.input + usage.output * p.output + usage.cache_read * p.cache_read
            + short_writes * p.cache_write
            + usage.cache_write_long * p.input * LONG_WRITE_TIMES_INPUT) / 1_000_000


def fill_cost(msg: Message, catalog_provider: str) -> Message:
    """Set usage.cost from the catalog when the provider didn't report one."""
    if msg.usage is not None and msg.usage.cost is None:
        msg.usage.cost = cost(msg.usage, lookup(catalog_provider, msg.model or ""))
    return msg


def providers_table() -> dict[str, dict[str, Any]]:
    """The compat table: how to reach each provider and its quirks (data/providers.json)."""
    return {k: v for k, v in _read("providers.json").items() if not k.startswith("_")}
