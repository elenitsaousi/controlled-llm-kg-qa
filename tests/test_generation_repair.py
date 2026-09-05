from kg.schema import KGSchema
from llm.candidate_generation import _candidate_key, _repair_generated_candidates


class FakeRepairClient:
    def generate(self, prompt, k=1):
        return [
            "SELECT ?regionName WHERE { "
            "?entry a survey:DemandForRegion ; "
            "survey:inRegion ?region . "
            "?region survey:regionName ?regionName . "
            "}"
        ]


def test_repair_generated_candidates_keeps_valid_repair_only():
    schema = KGSchema(
        {
            "classes": ["DemandForRegion", "Region"],
            "predicates": ["inRegion", "regionName"],
        }
    )
    candidates = [{"query": "not sparql", "source": "infineon"}]
    seen = {_candidate_key(candidates[0]["query"])}

    repaired = _repair_generated_candidates(
        question="Show demand by region.",
        schema=schema,
        candidates=candidates,
        seen=seen,
        client=FakeRepairClient(),
        max_repairs=1,
    )

    assert len(repaired) == 1
    assert repaired[0]["source"] == "repair"
    assert "DemandForRegion" in repaired[0]["query"]
