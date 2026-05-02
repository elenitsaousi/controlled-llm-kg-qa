#!/usr/bin/env python3
"""
Generate a 100-question Infineon benchmark with:
- natural-language question
- gold SPARQL query
- ambiguity label (low / mid / high)

The generator validates every query against the Infineon graph and keeps only
queries that execute successfully with non-empty results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from rdflib import Graph


PREFIX = """\
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

BASE_DIR = Path(__file__).resolve().parents[2]
GRAPH_PATH = BASE_DIR / "data" / "infineon" / "graph.ttl"
SEED_DATASET_PATH = BASE_DIR / "data" / "infineon" / "archive" / "infineon_dataset_30.json"
OUT_PATH = BASE_DIR / "data" / "infineon" / "infineon_train_generated.json"

TARGET_BY_LABEL = {"low": 34, "mid": 33, "high": 33}

SURVEYS: Sequence[Tuple[str, str]] = (
    ("Tier1", "Tier1_Survey"),
    ("OEM", "OEM_Survey"),
    ("Semiconductor", "Semiconductor_Survey"),
)


def _lit(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _normalize_query(q: str) -> str:
    return " ".join(q.strip().split())


def _query_values(graph: Graph, query: str) -> List[str]:
    rows = list(graph.query(PREFIX + query))
    values: List[str] = []
    for row in rows:
        if not row:
            continue
        values.append(str(row[0]))
    return values


def _regions(graph: Graph) -> List[str]:
    q = """
    SELECT DISTINCT ?name WHERE {
      ?r a survey:Region ;
         survey:regionName ?name .
    } ORDER BY ?name
    """
    return _query_values(graph, q)


@dataclass
class CandidateItem:
    label: str
    question: str
    query: str
    topic: str


class Builder:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.items: List[Dict[str, str]] = []
        self.counts: Dict[str, int] = {"low": 0, "mid": 0, "high": 0}
        self._seen: set[Tuple[str, str]] = set()
        self.skipped: List[str] = []

    def add(self, cand: CandidateItem) -> bool:
        if cand.label not in TARGET_BY_LABEL:
            self.skipped.append(f"invalid_label::{cand.label}::{cand.question}")
            return False
        if self.counts[cand.label] >= TARGET_BY_LABEL[cand.label]:
            return False

        key = (cand.question.strip().lower(), _normalize_query(cand.query))
        if key in self._seen:
            return False

        try:
            rows = list(self.graph.query(PREFIX + cand.query))
        except Exception as exc:  # pragma: no cover - runtime validation path
            self.skipped.append(f"invalid::{cand.question}::{exc}")
            return False

        if not rows:
            self.skipped.append(f"empty::{cand.question}")
            return False

        self.counts[cand.label] += 1
        qid = f"{cand.label.upper()}{self.counts[cand.label]}"
        self.items.append(
            {
                "id": qid,
                "question": cand.question,
                "query": cand.query.strip(),
                "ambiguity_label": cand.label,
                "topic": cand.topic,
            }
        )
        self._seen.add(key)
        return True

    def needs(self, label: str) -> int:
        return TARGET_BY_LABEL[label] - self.counts[label]


def _generate_low(builder: Builder, regions: Sequence[str]) -> None:
    # A) global counts/lists
    low_items = [
        CandidateItem(
            "low",
            "How many distinct regions exist in the Infineon survey graph?",
            "SELECT (COUNT(DISTINCT ?r) AS ?count) WHERE { ?r a survey:Region }",
            "catalog",
        ),
        CandidateItem(
            "low",
            "List all region names in the Infineon graph.",
            "SELECT ?name WHERE { ?r a survey:Region ; survey:regionName ?name } ORDER BY ?name",
            "catalog",
        ),
        CandidateItem(
            "low",
            "How many DemandForRegion entries exist in total?",
            "SELECT (COUNT(?d) AS ?count) WHERE { ?d a survey:DemandForRegion }",
            "demand",
        ),
        CandidateItem(
            "low",
            "How many Company instances are present?",
            "SELECT (COUNT(?c) AS ?count) WHERE { ?c a survey:Company }",
            "company",
        ),
        CandidateItem(
            "low",
            "How many Quarter entities exist?",
            "SELECT (COUNT(?q) AS ?count) WHERE { ?q a survey:Quarter }",
            "time",
        ),
        CandidateItem(
            "low",
            "How many FutureDemandAnalysis entries exist overall?",
            "SELECT (COUNT(?f) AS ?count) WHERE { ?f a survey:FutureDemandAnalysis }",
            "demand",
        ),
        CandidateItem(
            "low",
            "How many OrderCancellation entries exist overall?",
            "SELECT (COUNT(?o) AS ?count) WHERE { ?o a survey:OrderCancellation }",
            "orders",
        ),
        CandidateItem(
            "low",
            "How many AutonomousDrivingDevelopment entries exist?",
            "SELECT (COUNT(?a) AS ?count) WHERE { ?a a survey:AutonomousDrivingDevelopment }",
            "autonomous",
        ),
        CandidateItem(
            "low",
            "How many VehicleSalesObservation records exist?",
            "SELECT (COUNT(?v) AS ?count) WHERE { ?v a survey:VehicleSalesObservation }",
            "sales",
        ),
        CandidateItem(
            "low",
            "How many TechnologyCategory entities exist?",
            "SELECT (COUNT(?t) AS ?count) WHERE { ?t a survey:TechnologyCategory }",
            "technology",
        ),
        CandidateItem(
            "low",
            "How many SAE levels are represented?",
            "SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE { ?x survey:hasSAELevel ?s }",
            "autonomous",
        ),
        CandidateItem(
            "low",
            "List all baseline types used in the dataset.",
            "SELECT DISTINCT ?baseline WHERE { ?x survey:baselineType ?baseline } ORDER BY ?baseline",
            "baseline",
        ),
        CandidateItem(
            "low",
            "List all quarter labels used in the dataset.",
            "SELECT DISTINCT ?label WHERE { ?q a survey:Quarter ; survey:periodLabel ?label } ORDER BY ?label",
            "time",
        ),
        CandidateItem(
            "low",
            "List all technology category names.",
            "SELECT DISTINCT ?name WHERE { ?t a survey:TechnologyCategory ; survey:technologyCategoryName ?name } ORDER BY ?name",
            "technology",
        ),
        CandidateItem(
            "low",
            "Which survey types are present among Tier1, OEM and Semiconductor?",
            "SELECT DISTINCT ?type WHERE { ?s a ?type . FILTER(?type IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey)) } ORDER BY ?type",
            "survey",
        ),
    ]
    for item in low_items:
        builder.add(item)

    # B) survey-specific counts
    for survey_label, survey_cls in SURVEYS:
        builder.add(
            CandidateItem(
                "low",
                f"How many DemandForRegion entries are linked to {survey_label} survey?",
                f"""SELECT (COUNT(?d) AS ?count) WHERE {{
                ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o .
                ?o a survey:{survey_cls}
                }}""",
                "demand",
            )
        )
        builder.add(
            CandidateItem(
                "low",
                f"How many company entries are linked to {survey_label} survey?",
                f"""SELECT (COUNT(?c) AS ?count) WHERE {{
                ?c a survey:Company ; survey:hasSurveyOrigin ?o .
                ?o a survey:{survey_cls}
                }}""",
                "company",
            )
        )
        builder.add(
            CandidateItem(
                "low",
                f"How many FutureDemandAnalysis entries are linked to {survey_label} survey?",
                f"""SELECT (COUNT(?f) AS ?count) WHERE {{
                ?f a survey:FutureDemandAnalysis ; survey:hasSurveyOrigin ?o .
                ?o a survey:{survey_cls}
                }}""",
                "demand",
            )
        )
        builder.add(
            CandidateItem(
                "low",
                f"How many OrderCancellation entries are linked to {survey_label} survey?",
                f"""SELECT (COUNT(?oc) AS ?count) WHERE {{
                ?oc a survey:OrderCancellation ; survey:hasSurveyOrigin ?o .
                ?o a survey:{survey_cls}
                }}""",
                "orders",
            )
        )
        builder.add(
            CandidateItem(
                "low",
                f"How many {survey_label} companies report semiconductor shortage?",
                f"""SELECT (COUNT(?c) AS ?count) WHERE {{
                ?c a survey:Company ; survey:hasSurveyOrigin ?o ; survey:reportsShortage ?flag .
                ?o a survey:{survey_cls} .
                FILTER(?flag = true)
                }}""",
                "shortage",
            )
        )
        builder.add(
            CandidateItem(
                "low",
                f"How many {survey_label} companies do not report semiconductor shortage?",
                f"""SELECT (COUNT(?c) AS ?count) WHERE {{
                ?c a survey:Company ; survey:hasSurveyOrigin ?o ; survey:reportsShortage ?flag .
                ?o a survey:{survey_cls} .
                FILTER(?flag = false)
                }}""",
                "shortage",
            )
        )

    # C) per-region filtered counts (used as final low fillers)
    for survey_label, survey_cls in SURVEYS:
        if builder.needs("low") <= 0:
            break
        for region in regions:
            if builder.needs("low") <= 0:
                break
            builder.add(
                CandidateItem(
                    "low",
                    f"How many {survey_label} demand entries exist for region {region}?",
                    f"""SELECT (COUNT(?d) AS ?count) WHERE {{
                    ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:inRegion ?r .
                    ?o a survey:{survey_cls} .
                    ?r survey:regionName {_lit(region)}
                    }}""",
                    "demand",
                )
            )


def _generate_mid(builder: Builder, regions: Sequence[str]) -> None:
    for survey_label, survey_cls in SURVEYS:
        builder.add(
            CandidateItem(
                "mid",
                f"What is the total demand by region for {survey_label} survey?",
                f"""SELECT ?regionName (SUM(?units) AS ?totalDemand) WHERE {{
                ?d a survey:DemandForRegion ;
                   survey:hasSurveyOrigin ?o ;
                   survey:inRegion ?r ;
                   survey:totalDemand ?units .
                ?o a survey:{survey_cls} .
                ?r survey:regionName ?regionName .
                }} GROUP BY ?regionName ORDER BY DESC(?totalDemand)""",
                "demand",
            )
        )

    for survey_label, survey_cls in SURVEYS:
        for region in regions:
            builder.add(
                CandidateItem(
                    "mid",
                    f"What is the total {survey_label} demand for region {region}?",
                    f"""SELECT ?regionName (SUM(?units) AS ?totalDemand) WHERE {{
                    ?d a survey:DemandForRegion ;
                       survey:hasSurveyOrigin ?o ;
                       survey:inRegion ?r ;
                       survey:totalDemand ?units .
                    ?o a survey:{survey_cls} .
                    ?r survey:regionName ?regionName .
                    FILTER(?regionName = {_lit(region)})
                    }} GROUP BY ?regionName""",
                    "demand",
                )
            )

    for survey_label, survey_cls in SURVEYS:
        for region in regions:
            builder.add(
                CandidateItem(
                    "mid",
                    f"What is the average {survey_label} demand for region {region}?",
                    f"""SELECT ?regionName (AVG(?units) AS ?avgDemand) WHERE {{
                    ?d a survey:DemandForRegion ;
                       survey:hasSurveyOrigin ?o ;
                       survey:inRegion ?r ;
                       survey:totalDemand ?units .
                    ?o a survey:{survey_cls} .
                    ?r survey:regionName ?regionName .
                    FILTER(?regionName = {_lit(region)})
                    }} GROUP BY ?regionName""",
                    "demand",
                )
            )


def _generate_high(builder: Builder) -> None:
    for survey_label, survey_cls in SURVEYS:
        builder.add(
            CandidateItem(
                "high",
                f"How does {survey_label} demand percentage change by quarter and region?",
                f"""SELECT ?quarterLabel ?regionName (SUM(?pct) AS ?totalPctChange) WHERE {{
                ?d a survey:DemandForRegion ;
                   survey:hasSurveyOrigin ?o ;
                   survey:inRegion ?r ;
                   survey:quarter ?q ;
                   survey:totalDemandPercentageChange ?pct .
                ?o a survey:{survey_cls} .
                ?r survey:regionName ?regionName .
                ?q survey:periodLabel ?quarterLabel .
                }} GROUP BY ?quarterLabel ?regionName ORDER BY ?quarterLabel ?regionName""",
                "demand",
            )
        )
        builder.add(
            CandidateItem(
                "high",
                f"What is the quarterly demand percentage trend for {survey_label} survey?",
                f"""SELECT ?quarterLabel (AVG(?pct) AS ?avgPctChange) WHERE {{
                ?d a survey:DemandForRegion ;
                   survey:hasSurveyOrigin ?o ;
                   survey:quarter ?q ;
                   survey:totalDemandPercentageChange ?pct .
                ?o a survey:{survey_cls} .
                ?q survey:periodLabel ?quarterLabel .
                }} GROUP BY ?quarterLabel ORDER BY ?quarterLabel""",
                "demand",
            )
        )
        builder.add(
            CandidateItem(
                "high",
                f"Which region has the highest total demand in {survey_label} survey?",
                f"""SELECT ?regionName (SUM(?units) AS ?totalDemand) WHERE {{
                ?d a survey:DemandForRegion ;
                   survey:hasSurveyOrigin ?o ;
                   survey:inRegion ?r ;
                   survey:totalDemand ?units .
                ?o a survey:{survey_cls} .
                ?r survey:regionName ?regionName .
                }} GROUP BY ?regionName ORDER BY DESC(?totalDemand) LIMIT 1""",
                "demand",
            )
        )
        builder.add(
            CandidateItem(
                "high",
                f"Which region has the lowest total demand in {survey_label} survey?",
                f"""SELECT ?regionName (SUM(?units) AS ?totalDemand) WHERE {{
                ?d a survey:DemandForRegion ;
                   survey:hasSurveyOrigin ?o ;
                   survey:inRegion ?r ;
                   survey:totalDemand ?units .
                ?o a survey:{survey_cls} .
                ?r survey:regionName ?regionName .
                }} GROUP BY ?regionName ORDER BY ASC(?totalDemand) LIMIT 1""",
                "demand",
            )
        )
        builder.add(
            CandidateItem(
                "high",
                f"What is the shortage split (yes/no) for {survey_label} companies?",
                f"""SELECT ?status (COUNT(?c) AS ?count) WHERE {{
                ?c a survey:Company ;
                   survey:hasSurveyOrigin ?o ;
                   survey:reportsShortage ?flag .
                ?o a survey:{survey_cls} .
                BIND(IF(?flag = true, "yes", "no") AS ?status)
                }} GROUP BY ?status ORDER BY ?status""",
                "shortage",
            )
        )
        builder.add(
            CandidateItem(
                "high",
                f"What are total and average demand levels for {survey_label} survey?",
                f"""SELECT (SUM(?units) AS ?totalDemand) (AVG(?units) AS ?avgDemand) WHERE {{
                ?d a survey:DemandForRegion ;
                   survey:hasSurveyOrigin ?o ;
                   survey:totalDemand ?units .
                ?o a survey:{survey_cls}
                }}""",
                "demand",
            )
        )

    # Autonomous driving (Tier1 + OEM)
    for survey_label, survey_cls, root_cls in (
        ("Tier1", "Tier1_Survey", "AutonomousDrivingDevelopment_Tier1"),
        ("OEM", "OEM_Survey", "AutonomousDrivingDevelopment_OEM"),
    ):
        builder.add(
            CandidateItem(
                "high",
                f"What is the average autonomous-driving development for {survey_label} by vehicle type, SAE level and year?",
                f"""SELECT ?vehicle ?sae ?year (AVG(?pct) AS ?avgPct) WHERE {{
                ?root a survey:{root_cls} ;
                      survey:hasSurveyOrigin survey:{survey_cls} ;
                      survey:hasDetail ?entry .
                ?entry a survey:AutonomousDrivingDevelopment ;
                       survey:hasVehicleType ?vehicle ;
                       survey:hasSAELevel ?sae ;
                       survey:hasPercentage ?pct ;
                       survey:hasYear ?year .
                }} GROUP BY ?vehicle ?sae ?year ORDER BY ?vehicle ?sae ?year""",
                "autonomous",
            )
        )

    # Semiconductor: future-demand and order-cancellation analytics
    builder.add(
        CandidateItem(
            "high",
            "How does semiconductor future demand vary by technology category, quarter and baseline?",
            """SELECT ?tech ?quarterLabel ?baseline (AVG(?pct) AS ?avgPct) WHERE {
            ?f a survey:FutureDemandAnalysis ;
               survey:hasSurveyOrigin ?o ;
               survey:analyzesTechnologyCategory ?tech ;
               survey:forTimePeriod ?q ;
               survey:baselineType ?baseline ;
               survey:percentageChange ?pct .
            ?o a survey:Semiconductor_Survey .
            ?q survey:periodLabel ?quarterLabel .
            } GROUP BY ?tech ?quarterLabel ?baseline
              ORDER BY ?tech ?quarterLabel ?baseline""",
            "technology",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "What are semiconductor order-cancellation response counts by technology category and response type?",
            """SELECT ?tech ?responseType (SUM(xsd:integer(?cnt)) AS ?responses) WHERE {
            ?oc a survey:OrderCancellation ;
                survey:forTechnologyCategory ?tech ;
                survey:hasResponseType ?responseType ;
                survey:participantCount ?cnt .
            } GROUP BY ?tech ?responseType ORDER BY ?tech ?responseType""",
            "orders",
        )
    )

    # Cross-survey comparisons
    builder.add(
        CandidateItem(
            "high",
            "Compare total demand across Tier1, OEM and Semiconductor by region.",
            """SELECT ?surveyType ?regionName (SUM(?units) AS ?totalDemand) WHERE {
            ?d a survey:DemandForRegion ;
               survey:hasSurveyOrigin ?o ;
               survey:inRegion ?r ;
               survey:totalDemand ?units .
            ?o a ?surveyType .
            ?r survey:regionName ?regionName .
            FILTER(?surveyType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey))
            } GROUP BY ?surveyType ?regionName ORDER BY ?surveyType ?regionName""",
            "comparison",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "Compare company shortage status across Tier1, OEM and Semiconductor surveys.",
            """SELECT ?surveyType ?status (COUNT(?c) AS ?count) WHERE {
            ?c a survey:Company ;
               survey:hasSurveyOrigin ?o ;
               survey:reportsShortage ?flag .
            ?o a ?surveyType .
            FILTER(?surveyType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey))
            BIND(IF(?flag = true, "yes", "no") AS ?status)
            } GROUP BY ?surveyType ?status ORDER BY ?surveyType ?status""",
            "comparison",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "How many demand entries does each survey have per quarter label?",
            """SELECT ?surveyType ?quarterLabel (COUNT(?d) AS ?entries) WHERE {
            ?d a survey:DemandForRegion ;
               survey:hasSurveyOrigin ?o ;
               survey:quarter ?q .
            ?o a ?surveyType .
            ?q survey:periodLabel ?quarterLabel .
            FILTER(?surveyType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey))
            } GROUP BY ?surveyType ?quarterLabel ORDER BY ?surveyType ?quarterLabel""",
            "comparison",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "What is the average demand per survey type across all regions?",
            """SELECT ?surveyType (AVG(?units) AS ?avgDemand) WHERE {
            ?d a survey:DemandForRegion ;
               survey:hasSurveyOrigin ?o ;
               survey:totalDemand ?units .
            ?o a ?surveyType .
            FILTER(?surveyType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey))
            } GROUP BY ?surveyType ORDER BY ?surveyType""",
            "comparison",
        )
    )

    # Tier1 BL1 vs BL2
    builder.add(
        CandidateItem(
            "high",
            "What are Tier1 automotive percentage changes for baselines BL1 and BL2?",
            """SELECT ?baseline ?pct WHERE {
            survey:Tier1CurrentDemand survey:hasAggregatedResult ?entry .
            ?entry survey:baselineType ?baseline ;
                   survey:percentageChange ?pct .
            FILTER(?baseline IN ("BL1", "BL2"))
            } ORDER BY ?baseline""",
            "baseline",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "What is the BL1-BL2 delta for Tier1 current demand percentage change?",
            """SELECT
            (SUM(IF(?baseline = "BL1", ?pct, 0)) AS ?bl1)
            (SUM(IF(?baseline = "BL2", ?pct, 0)) AS ?bl2)
            ((SUM(IF(?baseline = "BL1", ?pct, 0)) - SUM(IF(?baseline = "BL2", ?pct, 0))) AS ?delta)
            WHERE {
            survey:Tier1CurrentDemand survey:hasAggregatedResult ?entry .
            ?entry survey:baselineType ?baseline ;
                   survey:percentageChange ?pct .
            FILTER(?baseline IN ("BL1", "BL2"))
            }""",
            "baseline",
        )
    )

    # Inventory and sales analytics
    builder.add(
        CandidateItem(
            "high",
            "How are Tier1 inventory responses distributed by component and trend?",
            """SELECT ?component ?trend (SUM(xsd:integer(?cnt)) AS ?responses) WHERE {
            ?entry a survey:InventoryDevelopment_Tier1 ;
                   survey:forComponent ?component ;
                   survey:inventoryTrend ?trend ;
                   survey:participantCount ?cnt .
            } GROUP BY ?component ?trend ORDER BY ?component ?trend""",
            "inventory",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "How are semiconductor inventory target indicators distributed by status?",
            """SELECT ?status (SUM(xsd:integer(?cnt)) AS ?responses) WHERE {
            ?entry a survey:InventoryTargetIndicator_Semi ;
                   survey:targetIndicatorStatus ?status ;
                   survey:participantCount ?cnt .
            } GROUP BY ?status ORDER BY ?status""",
            "inventory",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "For forecast data, what are units sold by vehicle type and period?",
            """SELECT ?vehicle ?period (SUM(?units) AS ?totalUnits) WHERE {
            ?obs a survey:VehicleSalesObservation ;
                 survey:hasVehicleType ?vehicle ;
                 survey:forTimePeriod ?period ;
                 survey:unitsSold ?units ;
                 survey:isForecastData true .
            } GROUP BY ?vehicle ?period ORDER BY ?vehicle ?period""",
            "sales",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "For actual data, what are units sold by vehicle type and period?",
            """SELECT ?vehicle ?period (SUM(?units) AS ?totalUnits) WHERE {
            ?obs a survey:VehicleSalesObservation ;
                 survey:hasVehicleType ?vehicle ;
                 survey:forTimePeriod ?period ;
                 survey:unitsSold ?units ;
                 survey:isActualData true .
            } GROUP BY ?vehicle ?period ORDER BY ?vehicle ?period""",
            "sales",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "For forecast data, what are total units sold by technology category?",
            """SELECT ?tech (SUM(?units) AS ?totalUnits) WHERE {
            ?obs a survey:VehicleSalesObservation ;
                 survey:hasTechnologyCategory ?tech ;
                 survey:unitsSold ?units ;
                 survey:isForecastData true .
            } GROUP BY ?tech ORDER BY DESC(?totalUnits)""",
            "sales",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "Which nameplates have the highest forecast units sold?",
            """SELECT ?nameplate (SUM(?units) AS ?totalUnits) WHERE {
            ?obs a survey:VehicleSalesObservation ;
                 survey:nameplateLabel ?nameplate ;
                 survey:unitsSold ?units ;
                 survey:isForecastData true .
            } GROUP BY ?nameplate ORDER BY DESC(?totalUnits) LIMIT 10""",
            "sales",
        )
    )

    builder.add(
        CandidateItem(
            "high",
            "How many assumptions exist per assigned vehicle type and technology category?",
            """SELECT ?vehicleType ?technologyCategory (COUNT(?a) AS ?count) WHERE {
            ?a survey:assignedVehicleType ?vehicleType ;
               survey:assignedTechnologyCategory ?technologyCategory .
            } GROUP BY ?vehicleType ?technologyCategory
              ORDER BY ?vehicleType ?technologyCategory""",
            "assumptions",
        )
    )


def _fallback_from_seed(builder: Builder) -> None:
    with open(SEED_DATASET_PATH, "r", encoding="utf-8") as f:
        seed = json.load(f)

    def label_for_query(query: str) -> str:
        upper = query.upper()
        complexity = 0
        complexity += upper.count("GROUP BY")
        complexity += upper.count("ORDER BY")
        complexity += upper.count("UNION")
        complexity += upper.count("BIND(")
        complexity += upper.count("FILTER(")
        if complexity <= 1:
            return "low"
        if complexity <= 3:
            return "mid"
        return "high"

    # Fallback only fills missing slots and keeps questions Infineon-specific.
    variant_idx = 1
    while True:
        pending = sum(max(0, builder.needs(lbl)) for lbl in TARGET_BY_LABEL)
        if pending <= 0:
            break
        progressed = False
        for item in seed:
            lbl = label_for_query(item["query"])
            if builder.needs(lbl) <= 0:
                continue
            question = f"[Infineon benchmark variant {variant_idx}] {item['question']}"
            cand = CandidateItem(lbl, question, item["query"], "seed_fallback")
            if builder.add(cand):
                progressed = True
                variant_idx += 1
            if sum(max(0, builder.needs(x)) for x in TARGET_BY_LABEL) <= 0:
                break
        if not progressed:
            # If we cannot make progress, stop to avoid infinite loop.
            break


def main() -> None:
    g = Graph()
    g.parse(str(GRAPH_PATH), format="turtle")

    regions = _regions(g)
    if not regions:
        raise RuntimeError("Could not load regions from Infineon graph.")

    builder = Builder(g)
    _generate_low(builder, regions)
    _generate_mid(builder, regions)
    _generate_high(builder)
    _fallback_from_seed(builder)

    total = len(builder.items)
    expected = sum(TARGET_BY_LABEL.values())
    if total != expected:
        raise RuntimeError(
            f"Dataset generation incomplete: got {total}, expected {expected}. "
            f"Counts={builder.counts}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(builder.items, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Saved {total} Infineon benchmark items to {OUT_PATH}")
    print(f"Counts by label: {builder.counts}")
    print(f"Skipped candidates: {len(builder.skipped)}")


if __name__ == "__main__":
    main()
