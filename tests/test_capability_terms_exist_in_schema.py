"""Regression guard: every class/predicate/property name referenced in
kg/capabilities.py's CapabilitySpec/DimensionSpec definitions should exist in
the real schema (data/infineon/schema.json).

An audit of the "graph_and_schema_metadata" question category found several
capability specs referencing plausible-looking but nonexistent term names
(e.g. "DemandForQuarter", "VehicleSales"/"VehicleSalesForecast" instead of
the real "VehicleSalesObservation", "monthLabel" instead of the real
"periodLabel", "ShortageStatus", "ComponentType", "InventoryTrend",
"InventoryResponse", "OrderCancellationResponse", a bare "year"). These were
harmless today only because capability/dimension matching ORs several terms
together and a real term elsewhere in the same tuple usually still matched
-- but that's an accident, not a guarantee, and it's dead/misleading
metadata that a future change could easily start relying on. This test
would have caught all of them.

A handful of legitimate non-schema tokens are expected (aggregation
functions, individual instance names like "OEM_Survey_Instance", SPARQL
variable-name hints) -- those are explicitly allow-listed below rather than
silently ignored, so a new genuine gap doesn't slip in unnoticed either.
"""

import json
from pathlib import Path

from kg.capabilities import DEFAULT_REGISTRY

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "infineon" / "schema.json"

# Not schema class/predicate/property names, but legitimately referenced:
# aggregation function names, and query-variable-name hints rather than
# literal schema terms.
_NON_SCHEMA_ALLOWLIST = {
    "AVG",
    "COUNT",
    "MAX",
    "MIN",
    "SUM",
    "responseType",
}


def _schema_names() -> set:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    names = set()
    names.update(schema.get("classes") or [])
    names.update(schema.get("predicates") or [])
    names.update(schema.get("properties") or [])
    return names


def test_every_required_and_core_term_exists_in_schema_or_is_allowlisted():
    schema_names = _schema_names()
    missing = []
    for capability in DEFAULT_REGISTRY.capabilities:
        for term in capability.core_terms:
            if term not in schema_names and term not in _NON_SCHEMA_ALLOWLIST:
                missing.append((capability.name, "core_term", term))
        for dimension in capability.dimensions:
            for term in dimension.required_terms:
                if term not in schema_names and term not in _NON_SCHEMA_ALLOWLIST:
                    missing.append((capability.name, f"dimension:{dimension.name}", term))

    assert not missing, (
        "capability/dimension terms not found in schema.json (fix the term or "
        f"add a justified allowlist entry): {missing}"
    )
