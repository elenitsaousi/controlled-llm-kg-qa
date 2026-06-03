#!/usr/bin/env python3
"""Use the configured LLM as a candidate selector on existing KGQA results.

This does not generate new SPARQL. It only asks the LLM to choose among the
already generated candidates, using compact query-plan summaries and hiding the
gold labels from the prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.apply_selection_to_results import _recompute_summary
from llm.client import InfineonGPTClient, LLMAuthError, LLMClientError
from ranking.feature_extraction import extract_query_plan
from ranking.query_contract import extract_question_contract


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _one_line(text: str, limit: int = 900) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _as_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    if value is None:
        return []
    return [str(value)]


def _plan_summary(query: str, schema: Dict[str, object]) -> Dict[str, object]:
    try:
        plan = extract_query_plan(query, schema)
    except Exception as exc:
        return {"error": str(exc)}
    keys = [
        "classes",
        "predicates",
        "survey_origins",
        "aggregations",
        "query_types",
        "group_by_vars",
        "group_by_predicates",
        "select_vars",
    ]
    return {key: _as_list(plan.get(key)) for key in keys if plan.get(key)}


def _format_plan(plan: Dict[str, object]) -> str:
    if not plan:
        return "no extracted plan"
    parts = []
    for key, values in plan.items():
        vals = _as_list(values)
        if vals:
            parts.append(f"{key}={', '.join(vals[:8])}")
    return "; ".join(parts) if parts else "no extracted plan"


def _candidate_prompt_block(
    idx: int,
    candidate: Dict[str, object],
    schema: Dict[str, object],
) -> str:
    query = str(candidate.get("query", "") or "")
    plan = _plan_summary(query, schema)
    answer = candidate.get("answer") or candidate.get("answer_preview") or candidate.get("execution_preview")
    row_count = candidate.get("execution_row_count") or candidate.get("row_count")
    extras = []
    if row_count is not None:
        extras.append(f"rows={row_count}")
    if answer:
        extras.append(f"answer_preview={_one_line(str(answer), 350)}")
    extra_text = "\n".join(extras) if extras else "no execution preview"
    return (
        f"Candidate {idx}\n"
        f"Plan: {_format_plan(plan)}\n"
        f"{extra_text}\n"
        f"SPARQL: {_one_line(query, 900)}"
    )


def _selector_prompt(
    question: str,
    candidates: Sequence[Dict[str, object]],
    schema: Dict[str, object],
) -> str:
    contract = extract_question_contract(question).to_dict()
    blocks = [
        _candidate_prompt_block(idx, cand, schema)
        for idx, cand in enumerate(candidates, start=1)
    ]
    return f"""You are selecting the best SPARQL candidate for an Infineon True Demand KGQA system.

Choose the candidate that most directly answers the user question.

Important selection rules:
- Match the requested aggregation exactly: SUM/total, AVG/average, COUNT/how many, ranking/highest/lowest.
- Match the requested metric: demand, future demand, current demand, inventory, vehicle sales, shortage, autonomous driving, order cancellation.
- Match requested scope/origin: OEM, Tier1, Semiconductor.
- Match requested dimensions/grouping: region, quarter, month, year, technology category, vehicle type, SAE level, component, trend, response type, baseline.
- Match filters such as actual vs forecast, BL1 vs BL2, shortage yes/no.
- Do not prefer a broader query if a candidate matches the requested scope and dimensions.
- If no candidate is clearly better, choose Candidate 1.

Return only JSON with this shape:
{{"choice": 1, "confidence": 0.0, "reason": "short reason"}}

User question:
{question}

Extracted question contract:
{json.dumps(contract, ensure_ascii=False, sort_keys=True)}

Candidates:
{chr(10).join(blocks)}
"""


CHOICE_RE = re.compile(r'"choice"\s*:\s*(\d+)|\bchoice\s*[:=]\s*(\d+)', re.IGNORECASE)
CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)|\bconfidence\s*[:=]\s*([0-9.]+)', re.IGNORECASE)


def _parse_selector_response(text: str, max_choice: int) -> Dict[str, object]:
    cleaned = str(text or "").strip()
    try:
        payload = json.loads(cleaned)
        choice = int(payload.get("choice", 1))
        confidence = float(payload.get("confidence", 0.0))
        reason = str(payload.get("reason", "") or "")
    except Exception:
        choice_match = CHOICE_RE.search(cleaned)
        conf_match = CONF_RE.search(cleaned)
        choice = int(next(g for g in choice_match.groups() if g)) if choice_match else 1
        confidence = float(next(g for g in conf_match.groups() if g)) if conf_match else 0.0
        reason = cleaned[:300]
    choice = min(max(choice, 1), max_choice)
    confidence = min(max(confidence, 0.0), 1.0)
    return {"choice": choice, "confidence": confidence, "reason": reason}


def _select_detail(
    detail: Dict[str, object],
    client: InfineonGPTClient,
    schema: Dict[str, object],
    top_n: int,
    min_confidence: float,
) -> Dict[str, object]:
    updated = deepcopy(detail)
    candidates = list(updated.get("candidates") or [])
    if len(candidates) < 2:
        return updated

    question = str(updated.get("effective_question") or updated.get("question") or "")
    window = candidates[: max(2, int(top_n))]
    prompt = _selector_prompt(question, window, schema)
    response = client.generate_text(prompt)
    parsed = _parse_selector_response(response, len(window))
    choice_idx = int(parsed["choice"]) - 1
    confidence = float(parsed["confidence"])
    if choice_idx <= 0 or confidence < float(min_confidence):
        updated["llm_selector"] = {
            "applied": False,
            "choice": int(parsed["choice"]),
            "confidence": confidence,
            "reason": parsed.get("reason", ""),
            "raw": response,
        }
        return updated

    chosen = window[choice_idx]
    chosen_key = _query_key(str(chosen.get("query", "")))
    reordered = []
    used = False
    for cand in candidates:
        cand_key = _query_key(str(cand.get("query", "")))
        if not used and cand_key == chosen_key:
            cand_copy = deepcopy(cand)
            cand_copy["llm_selector_score"] = confidence
            reordered.insert(0, cand_copy)
            used = True
            continue
        else:
            reordered.append(deepcopy(cand))
    updated["candidates"] = reordered
    updated["llm_selector"] = {
        "applied": True,
        "choice": int(parsed["choice"]),
        "confidence": confidence,
        "reason": parsed.get("reason", ""),
        "raw": response,
    }
    return updated


def _completed_ids(payload: Dict[str, object]) -> set:
    return {
        str(detail.get("id"))
        for detail in payload.get("details", [])
        if isinstance(detail, dict) and detail.get("llm_selector") is not None
    }


def apply_llm_selector(
    results_path: str,
    schema_path: str,
    out_path: str,
    top_n: int = 4,
    min_confidence: float = 0.0,
    resume_from: Optional[str] = None,
    save_every: int = 1,
) -> Dict[str, object]:
    payload = _load_json(results_path)
    schema = _load_json(schema_path)
    original_details = list(payload.get("details") or [])

    resumed_by_id: Dict[str, Dict[str, object]] = {}
    if resume_from and Path(resume_from).exists():
        resumed = _load_json(resume_from)
        resumed_by_id = {
            str(detail.get("id")): detail
            for detail in resumed.get("details", [])
            if isinstance(detail, dict) and detail.get("llm_selector") is not None
        }

    client = InfineonGPTClient(temperature=0.0, max_tokens=300)
    details: List[Dict[str, object]] = []
    for idx, detail in enumerate(original_details, start=1):
        qid = str(detail.get("id"))
        if qid in resumed_by_id:
            details.append(resumed_by_id[qid])
            continue
        try:
            selected = _select_detail(
                detail,
                client,
                schema,
                top_n=top_n,
                min_confidence=min_confidence,
            )
        except (LLMAuthError, LLMClientError):
            partial = deepcopy(payload)
            partial["details"] = details + original_details[len(details) :]
            partial["summary"] = _recompute_summary(details, payload.get("summary") or {})
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, indent=2, ensure_ascii=False)
                f.write("\n")
            raise
        details.append(selected)
        if save_every and idx % int(save_every) == 0:
            partial = deepcopy(payload)
            partial["details"] = details + original_details[len(details) :]
            partial["summary"] = _recompute_summary(details, payload.get("summary") or {})
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(partial, f, indent=2, ensure_ascii=False)
                f.write("\n")

    updated = deepcopy(payload)
    updated["details"] = details
    updated["summary"] = _recompute_summary(details, payload.get("summary") or {})
    changed = 0
    for before, after in zip(original_details, details):
        before_candidates = before.get("candidates") or []
        after_candidates = after.get("candidates") or []
        if before_candidates and after_candidates and _query_key(str(before_candidates[0].get("query", ""))) != _query_key(str(after_candidates[0].get("query", ""))):
            changed += 1
    updated["llm_selector_rewrite"] = {
        "source_results": results_path,
        "schema": schema_path,
        "top_n": int(top_n),
        "min_confidence": float(min_confidence),
        "changed_count": changed,
        "note": "LLM selector chooses among existing candidates only; gold labels are not included in the prompt.",
    }
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an LLM-as-selector reranker to an existing evaluation JSON."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--resume-from")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    updated = apply_llm_selector(
        results_path=args.results,
        schema_path=args.schema,
        out_path=args.out,
        top_n=args.top_n,
        min_confidence=args.min_confidence,
        resume_from=args.resume_from,
        save_every=args.save_every,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")

    summary = updated["summary"]
    rewrite = updated["llm_selector_rewrite"]
    print("===== APPLY LLM SELECTOR TO RESULTS =====")
    print(f"Input: {args.results}")
    print(f"Changed selections: {rewrite['changed_count']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
