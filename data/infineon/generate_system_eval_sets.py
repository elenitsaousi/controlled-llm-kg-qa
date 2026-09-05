#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


BASE = Path(__file__).resolve().parent
ROUTING_OUT = BASE / "request_routing_eval.json"
CLARIFICATION_OUT = BASE / "clarification_behavior_eval.json"


def _rows(
    prefix: str,
    questions: List[str],
    *,
    start: int,
    **fields,
) -> List[Dict[str, object]]:
    return [
        {"id": f"{prefix}{idx:03d}", "question": question, **fields}
        for idx, question in enumerate(questions, start=start)
    ]


def build_routing_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rows += _rows(
        "ROUTE",
        [
            "What is OrderCancellation?",
            "What is VehicleSalesObservation?",
            "What is CurrentDemandAnalysis?",
            "What is InventoryDevelopment?",
            "What is AutonomousDrivingDevelopment?",
            "What is ComponentShare?",
            "What does BL1 mean?",
            "What does BL2 mean?",
            "What does OEM mean?",
            "What is Tier1?",
            "What is hasSurveyOrigin?",
            "What is participantCount?",
            "What is baselineB1?",
            "Define ComponentType.",
            "Define InventoryTargetIndicator.",
        ],
        start=len(rows) + 1,
        expected_route="definition",
    )
    rows += _rows(
        "ROUTE",
        [
            "What is a manufacturer?",
            "What is a supplier?",
            "What is a response category?",
            "What is a dashboard?",
            "What does lead time mean?",
            "Define procurement.",
            "What is a forecast?",
            "What is a percentage?",
            "What is an observation?",
            "What is a benchmark?",
            "Define segmentation.",
            "What does aggregation mean?",
            "What is a table?",
            "What is a trend?",
            "Define an entity.",
        ],
        start=len(rows) + 1,
        expected_route="general_definition",
    )
    rows += _rows(
        "ROUTE",
        [
            "What about OrderCancellation?",
            "Tell me about InventoryDevelopment.",
            "Show me CurrentDemandAnalysis.",
            "What about VehicleSalesObservation?",
            "Tell me about ComponentShare.",
            "Show me AutonomousDrivingDevelopment.",
            "What about BL1?",
            "Tell me about Tier1.",
            "Show me Inventory.",
            "What about Company?",
        ],
        start=len(rows) + 1,
        expected_route="clarification_needed",
    )
    rows += _rows(
        "ROUTE",
        [
            "What is the weather tomorrow?",
            "Will it rain next week?",
            "Who is the president of France?",
            "What is Apple's stock price?",
            "Give me a pasta recipe.",
            "Who won the football match yesterday?",
            "What is the temperature in Berlin?",
            "Show me today's soccer schedule.",
            "What is the latest exchange rate?",
            "Recommend a restaurant nearby.",
        ],
        start=len(rows) + 1,
        expected_route="out_of_domain",
    )
    rows += _rows(
        "ROUTE",
        [
            "Return monthly totals for actual vehicle-sales observations.",
            "Which month has the highest actual vehicle sales?",
            "Give forecast vehicle-sales totals grouped by month.",
            "Show yearly autonomous-driving averages by vehicle type.",
            "At SAE Level 5, which vehicle type has the highest percentage?",
            "Group order-cancellation participant counts by technology category and response type.",
            "Which cancellation response is most common by technology category?",
            "Summarize shortage responses by survey origin.",
            "Show inventory trends by component.",
            "Compare BL1 and BL2 current-demand changes for Tier1 Automotive.",
        ],
        start=len(rows) + 1,
        expected_route="kg_query",
    )
    return rows


def build_clarification_rows() -> List[Dict[str, object]]:
    ambiguous = [
        ("Summarize cancellation responses across technology categories.", "order_cancellation"),
        ("What do cancellation responses look like by technology category?", "order_cancellation"),
        ("Show cancellation results across technologies.", "order_cancellation"),
        ("Describe order-cancellation behavior by technology.", "order_cancellation"),
        ("Compare cancellation responses for technology categories.", "order_cancellation"),
        ("Summarize shortage responses across survey origins.", "shortage"),
        ("What do shortage results look like by origin?", "shortage"),
        ("Show shortage behavior across survey groups.", "shortage"),
        ("Compare shortages across OEM, Tier1, and Semiconductor.", "shortage"),
        ("Describe shortage responses by survey.", "shortage"),
        ("Show inventory trends by component.", "inventory"),
        ("What does inventory look like across components?", "inventory"),
        ("Summarize inventory by component.", "inventory"),
        ("Compare inventory behavior across components.", "inventory"),
        ("Describe component inventory results.", "inventory"),
        ("Show autonomous-driving results by vehicle type.", "autonomous"),
        ("What does autonomous-driving development look like by vehicle?", "autonomous"),
        ("Compare autonomous-driving percentages across vehicle types.", "autonomous"),
        ("Summarize autonomous-driving results over years.", "autonomous"),
        ("Describe autonomous development by vehicle category.", "autonomous"),
        ("Show actual vehicle-sales results over time.", "vehicle_sales"),
        ("What do actual sales look like by month?", "vehicle_sales"),
        ("Compare actual vehicle sales across time periods.", "vehicle_sales"),
        ("Summarize monthly actual sales.", "vehicle_sales"),
        ("Describe actual vehicle-sales observations.", "vehicle_sales"),
        ("Show BL1 and BL2 current-demand results.", "current_demand"),
        ("What do the baseline changes look like for Tier1 Automotive?", "current_demand"),
        ("Summarize current-demand baselines for Tier1 Automotive.", "current_demand"),
        ("Compare baseline scenarios for Tier1 Automotive.", "current_demand"),
        ("Describe BL1 versus BL2 for Tier1 Automotive.", "current_demand"),
    ]
    explicit = [
        ("Group order-cancellation participant counts by technology category and response type.", "order_cancellation"),
        ("Which cancellation response is most common by technology category?", "order_cancellation"),
        ("Return participant totals by technology category and response type.", "order_cancellation"),
        ("Count order-cancellation entries by technology category.", "order_cancellation"),
        ("List cancellation response types by technology category.", "order_cancellation"),
        ("Summarize shortage totals by survey origin.", "shortage"),
        ("Count shortages by survey origin.", "shortage"),
        ("Return shortage values grouped by survey origin.", "shortage"),
        ("Which survey origin has the highest shortage total?", "shortage"),
        ("List shortage responses by survey origin.", "shortage"),
        ("Return inventory trends by component.", "inventory"),
        ("Count inventory observations by component.", "inventory"),
        ("List inventory trend values for each component.", "inventory"),
        ("Which component has decreasing inventory?", "inventory"),
        ("Show component inventory values.", "inventory"),
        ("Show yearly autonomous-driving averages by vehicle type.", "autonomous"),
        ("At SAE Level 5, which vehicle type has the highest percentage?", "autonomous"),
        ("Return autonomous-driving percentages by vehicle type and year.", "autonomous"),
        ("Count autonomous-driving entries by vehicle type.", "autonomous"),
        ("List SAE percentages for each vehicle type.", "autonomous"),
        ("Return monthly totals for actual vehicle-sales observations.", "vehicle_sales"),
        ("Which month has the highest actual vehicle sales?", "vehicle_sales"),
        ("Sum actual vehicle sales for each month.", "vehicle_sales"),
        ("Count actual vehicle-sales observations by month.", "vehicle_sales"),
        ("List actual sold units by time period.", "vehicle_sales"),
        ("Compare BL1 and BL2 current-demand changes for Tier1 Automotive.", "current_demand"),
        ("What are the BL1 and BL2 percentage changes for Tier1 Automotive?", "current_demand"),
        ("Return baseline-level current-demand percentages for Tier1 Automotive.", "current_demand"),
        ("List Tier1 Automotive current-demand changes for BL1 and BL2.", "current_demand"),
        ("Give the average BL1 and BL2 current-demand changes for Tier1 Automotive.", "current_demand"),
    ]
    rows: List[Dict[str, object]] = []
    for idx, (question, topic) in enumerate(ambiguous, start=1):
        rows.append(
            {
                "id": f"CLAR{idx:03d}",
                "question": question,
                "expected_needs_clarification": True,
                "topic": topic,
            }
        )
    for idx, (question, topic) in enumerate(explicit, start=len(rows) + 1):
        rows.append(
            {
                "id": f"CLAR{idx:03d}",
                "question": question,
                "expected_needs_clarification": False,
                "topic": topic,
            }
        )
    return rows


def main() -> None:
    routing = build_routing_rows()
    clarification = build_clarification_rows()
    ROUTING_OUT.write_text(json.dumps(routing, indent=2) + "\n", encoding="utf-8")
    CLARIFICATION_OUT.write_text(json.dumps(clarification, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(routing)} routing rows to {ROUTING_OUT}")
    print(f"Wrote {len(clarification)} clarification rows to {CLARIFICATION_OUT}")


if __name__ == "__main__":
    main()
