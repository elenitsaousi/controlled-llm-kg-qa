"""Regression guard for the region/survey-group "guided query" builders in
app.py (`_region_values_for_question`, `_survey_values_for_question`).

These functions turn free-text mentions ("Asia", "Tier1", "not Japan", ...)
into literal SPARQL entity references that get spliced straight into a
deterministic query -- there is no candidate ranking or answerability guard
downstream, so a wrong entity here means the query is wrong, full stop (see
the region-matching bug fixed in app.py: a dangling `survey:China` URI that
matched nothing, and "Asia" incorrectly pulling in Japan with no way to
exclude it).

Rather than re-testing only the two questions that surfaced that bug, this
walks every alias branch and the exclusion mechanism, and independently
verifies against the real graph that every URI these functions can produce
actually exists as a graph individual. This is meant to catch the same class
of bug (dangling/mismatched entity references, alias/taxonomy drift) for any
future question shape, not just the ones already reported.
"""

import re
from pathlib import Path

import pytest
from rdflib import Graph, URIRef

import app

GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "infineon" / "graph.ttl"

SURVEY_NS = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"


@pytest.fixture(scope="module")
def graph():
    g = Graph()
    g.parse(str(GRAPH_PATH), format="turtle")
    return g


def _to_uriref(token: str) -> URIRef:
    token = token.strip()
    if token.startswith("<") and token.endswith(">"):
        return URIRef(token[1:-1])
    assert token.startswith("survey:"), f"unexpected token shape: {token!r}"
    return URIRef(SURVEY_NS + token[len("survey:") :])


def _region_uris(q_norm: str):
    raw = app._region_values_for_question(q_norm)
    return [_to_uriref(line) for line in raw.splitlines() if line.strip()]


def _survey_uris(q_norm: str):
    raw = app._survey_values_for_question(q_norm)
    tokens = re.findall(r"survey:\S+(?=\s+\")", raw)
    return [_to_uriref(t) for t in tokens]


def _exists(graph, uri: URIRef) -> bool:
    return (uri, None, None) in graph or (None, None, uri) in graph


REGION_PROBES = [
    "show demand for americas",
    "show demand for america",
    "show demand for us",
    "show demand for europe",
    "show demand for japan",
    "show demand for china",
    "show demand for all other regions",
    "show demand for other regions",
    "show demand for asia",
    "show demand for asian markets",
    "show demand for asia pacific",
    "show demand for apac",
]


@pytest.mark.parametrize("question", REGION_PROBES)
def test_region_alias_resolves_to_real_graph_individual(graph, question):
    uris = _region_uris(question)
    assert uris, f"expected at least one region match for {question!r}"
    for uri in uris:
        assert _exists(graph, uri), f"{question!r} -> {uri} does not exist in the graph"


def test_asia_does_not_include_japan():
    # Japan is modeled as its own top-level region, a sibling of Asia
    # Pacific/China and Asia Pacific/All Other -- not a member of it. See
    # the WSTS-style region breakdown in data/infineon/graph.ttl.
    japan = _to_uriref("survey:RegionJapan")
    for question in ["show demand for asia", "show demand for asia pacific", "show demand for apac"]:
        assert japan not in _region_uris(question), (
            f"{question!r} should not resolve to Japan"
        )


@pytest.mark.parametrize(
    "question,excluded_alias",
    [
        ("show demand for asia but not china", "china"),
        ("show demand for asia excluding china", "china"),
        ("show demand for apac except china", "china"),
        ("show demand for europe and america but not europe", "europe"),
    ],
)
def test_region_exclusion_removes_the_named_region(question, excluded_alias):
    china = _to_uriref(
        "<http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/RegionAsiaPacific/China>"
    )
    europe = _to_uriref("survey:RegionEurope")
    excluded_uri = {"china": china, "europe": europe}[excluded_alias]
    assert excluded_uri not in _region_uris(question)


SURVEY_PROBES = [
    "show demand for oem",
    "show demand for oems",
    "show demand for tier1",
    "show demand for tier-1",
    "show demand for tier 1",
    "show demand for semi",
    "show demand for semis",
    "show demand for semiconductor",
    "show demand for semiconductors",
]


@pytest.mark.parametrize("question", SURVEY_PROBES)
def test_survey_group_alias_resolves_to_real_graph_individual(graph, question):
    uris = _survey_uris(question)
    assert uris, f"expected at least one survey-group match for {question!r}"
    for uri in uris:
        assert _exists(graph, uri), f"{question!r} -> {uri} does not exist in the graph"


def test_survey_group_with_no_match_falls_back_to_all_three(graph):
    uris = _survey_uris("show demand overall")
    assert len(uris) == 3
    for uri in uris:
        assert _exists(graph, uri)
