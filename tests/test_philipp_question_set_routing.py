"""Offline regression test for Philipp's question set (docs/philipp_true_demand_dr_test_questions.md).

Only checks the subset of questions whose expected_route is reliably
gradable via route_request() alone, without executing SPARQL against a live
graph (requires_live_fuseki: false in the fixture). Everything else needs
pipeline.qa.answer_question() run against a real Fuseki endpoint to grade
properly -- see evaluation/run_philipp_question_set.py for that.
"""

import json
from pathlib import Path

import pytest

from pipeline.request_routing import route_request

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "evaluation"
    / "question_sets"
    / "philipp_true_demand_dr_test_questions.json"
)


def _load_fixture():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


FIXTURE = _load_fixture()
ROUTABLE = [item for item in FIXTURE if not item["requires_live_fuseki"]]


def test_fixture_loads_and_has_expected_shape():
    assert len(FIXTURE) == 148
    for item in FIXTURE:
        assert item["question"].strip()
        assert item["category"]
        assert item["expected_route"]
        assert isinstance(item["requires_live_fuseki"], bool)
    ids = [item["id"] for item in FIXTURE]
    assert len(ids) == len(set(ids)), "fixture ids must be unique"


@pytest.mark.parametrize(
    "item",
    ROUTABLE,
    ids=[item["id"] for item in ROUTABLE],
)
def test_routable_question_matches_expected_route(item):
    actual = route_request(item["question"]).get("route")
    assert actual == item["expected_route"], (
        f"{item['id']} ({item['category']}): {item['question']!r} "
        f"expected route {item['expected_route']!r}, got {actual!r}"
    )
