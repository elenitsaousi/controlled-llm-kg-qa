#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


AUTONOMOUS_REPAIR_NOTE = (
    "Gold query used OEM/Tier1-specific autonomous-driving classes as instance types. "
    "The KG stores OEM/Tier1 autonomous-driving data as hasDetail links from the OEM/Tier1 "
    "class nodes to AutonomousDrivingDevelopment detail instances. Corrected path should return rows."
)

INVENTORY_REPAIR_NOTE = (
    "Gold query used an aggregate hasInventoryResponse/participantCount path that returns no rows. "
    "The KG stores Tier1 inventory observations as InventoryDevelopment_Tier1 entries with forCompany, "
    "forComponent, and inventoryTrend. Corrected path counts those row-level observations by component group."
)

INVENTORY_TIER1_COMPONENT_QUERY = """SELECT ?category
       (SUM(IF(?trend = "Increase"^^xsd:string, 1, 0)) AS ?increase)
       (SUM(IF(?trend = "Decrease"^^xsd:string, 1, 0)) AS ?decrease)
       (SUM(IF(?trend = "Stable"^^xsd:string, 1, 0)) AS ?stable)
       (COUNT(?entry) AS ?total)
WHERE {
  ?entry a survey:InventoryDevelopment_Tier1 ;
         survey:forCompany ?company ;
         survey:forComponent ?comp ;
         survey:inventoryTrend ?trend .
  BIND(
    IF(?comp = survey:Component_EV, "EV",
      IF(?comp = survey:Component_non_EV, "non EV",
        IF(?comp = survey:Component_both, "mixed", REPLACE(STRAFTER(STR(?comp), "Component_"), "_", " "))
      )
    ) AS ?category
  )
  BIND(IF(?category = "EV", 1, IF(?category = "non EV", 2, IF(?category = "mixed", 3, 4))) AS ?ord)
}
GROUP BY ?category ?ord
ORDER BY ?ord"""


def _load_empty_review_rows(path: str) -> Dict[str, Dict[str, str]]:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig")))
    return {
        str(row.get("id", "")): row
        for row in rows
        if str(row.get("needs_manual_review", "")).strip().lower() == "true"
    }


def _repair_autonomous_query(query: str) -> Tuple[str, bool]:
    pattern = re.compile(
        r"\?[A-Za-z_][A-Za-z0-9_]*\s+a\s+survey:AutonomousDrivingDevelopment_(OEM|Tier1)\s*;\s*"
        r"survey:hasSurveyOrigin\s+survey:(OEM|Tier1)_Survey\s*;\s*"
        r"survey:hasDetail\s+(\?[A-Za-z_][A-Za-z0-9_]*)\s*\.",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        scope = match.group(1)
        detail_var = match.group(3)
        return f"survey:AutonomousDrivingDevelopment_{scope} survey:hasDetail {detail_var} ."

    repaired, count = pattern.subn(repl, query)
    return repaired, count > 0


def repair_gold_dataset(gold_path: str, review_csv_path: str) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    gold_rows: List[Dict[str, object]] = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    empty_rows = _load_empty_review_rows(review_csv_path)
    counts = {
        "rows": len(gold_rows),
        "autonomous_repaired": 0,
        "inventory_repaired": 0,
        "unmodified_empty": 0,
    }

    for row in gold_rows:
        qid = str(row.get("id", ""))
        review = empty_rows.get(qid)
        if not review:
            continue
        family = str(review.get("expected_family", "") or row.get("family", ""))
        query = str(row.get("query", ""))
        if family == "autonomous_driving":
            repaired, changed = _repair_autonomous_query(query)
            if changed:
                row["query"] = repaired
                row["gold_repair_status"] = "repaired"
                row["gold_repair_note"] = AUTONOMOUS_REPAIR_NOTE
                counts["autonomous_repaired"] += 1
            else:
                counts["unmodified_empty"] += 1
        elif family == "inventory":
            row["query"] = INVENTORY_TIER1_COMPONENT_QUERY
            row["gold_repair_status"] = "repaired"
            row["gold_repair_note"] = INVENTORY_REPAIR_NOTE
            counts["inventory_repaired"] += 1
        else:
            counts["unmodified_empty"] += 1
    return gold_rows, counts


def write_repaired_review_csv(review_csv_path: str, out_csv: str) -> Dict[str, int]:
    rows = list(csv.DictReader(Path(review_csv_path).open(encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    counts = {"autonomous_marked": 0, "inventory_marked": 0}
    for row in rows:
        needs_review = str(row.get("needs_manual_review", "")).strip().lower() == "true"
        if not needs_review:
            continue
        family = str(row.get("expected_family", ""))
        if family == "autonomous_driving":
            row["human_gold_valid"] = "incorrect_repairable"
            row["human_notes"] = AUTONOMOUS_REPAIR_NOTE
            counts["autonomous_marked"] += 1
        elif family == "inventory":
            row["human_gold_valid"] = "incorrect_repairable"
            row["human_notes"] = INVENTORY_REPAIR_NOTE
            counts["inventory_marked"] += 1
    with Path(out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair systematic empty-result queries in the final1000 gold dataset.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--out-gold", required=True)
    parser.add_argument("--out-review-csv", required=True)
    parser.add_argument("--out-manifest", default="")
    args = parser.parse_args()

    repaired_rows, repair_counts = repair_gold_dataset(args.gold, args.review_csv)
    Path(args.out_gold).write_text(json.dumps(repaired_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review_counts = write_repaired_review_csv(args.review_csv, args.out_review_csv)
    manifest = {"repair_counts": repair_counts, "review_counts": review_counts}
    if args.out_manifest:
        Path(args.out_manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("===== GOLD DATASET EMPTY-QUERY REPAIR =====")
    print(json.dumps(manifest, indent=2))
    print(f"Repaired gold: {args.out_gold}")
    print(f"Updated review CSV: {args.out_review_csv}")
    if args.out_manifest:
        print(f"Manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
