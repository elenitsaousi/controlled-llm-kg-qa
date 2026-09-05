#!/usr/bin/env python3
"""Execution-grounded signature entropy (H_sig) for candidate SPARQL sets.

This is the diagnostic described in the thesis methodology (Section 5.3) but
never fully realized: earlier evaluation runs stored only row counts, not
per-candidate row previews, so ``analyze_execution_signature_entropy.py``
could only cluster candidates by execution signature when a preview already
happened to be present in the results file, and fell back to query-text
signatures otherwise (see the discussion in Chapter 7 on why H_sig was left
as a diagnostic rather than a quantitative result).

This script closes that gap by actually executing every candidate query
against the True Demand graph (local RDFLib or Fuseki), with a short
per-query timeout, and building the execution signature from the query's own
result shape rather than from whatever happened to be logged earlier:

    signature(q_i) = (selected variables, row count, normalized preview rows)

Candidates within a question are grouped by this signature, their
softmax-normalized scores are summed per group, and entropy is computed over
the resulting group probabilities:

    H_sig(Q)      = -sum_k P_k * log(P_k)                (natural log, nats)
    H_sig_norm(Q) = H_sig(Q) / log(m)                     (m = signature groups)

Both values are written per-question to CSV and JSON, alongside the
unclustered (raw, per-candidate) entropy for comparison.

Two clustering variants are computed side by side, because a naive
implementation can be over-credited: if two candidates both time out, both
error, or both simply return zero rows, they are not "the same answer" in
any meaningful sense -- they just both failed or matched nothing. Grouping
those together would mix genuine execution-equivalence with coincidental
shared failure and inflate the apparent entropy reduction.

  - H_sig / H_sig_norm (lenient): all timeouts for a question share one
    signature, and all 0-row successes with the same projected variables
    share one signature, alongside genuine non-empty result equivalence.
  - H_sig_strict / H_sig_strict_norm: timeouts, errors, and 0-row results
    are never clustered with one another -- each keeps a unique signature
    -- so only candidates producing the exact same non-empty result are
    ever grouped. This is the conservative number to cite when the claim
    needs to survive a reviewer asking exactly that question.

Input format: a JSON file shaped like the existing selection/analysis result
files under results/ (e.g. final1000_repaired_holdout200_xgb_pairwise_ltr_selection.json),
i.e. {"details": [{"id": ..., "question": ..., "candidates": [{"query": ...,
"ml_score": ...}, ...]}, ...]}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(str(PROJECT_ROOT / ".env"))
    load_dotenv()
except Exception:
    pass

try:
    from rdflib import Graph
except ImportError:  # pragma: no cover - allows --help without rdflib installed.
    Graph = None  # type: ignore[assignment]

try:
    from kg.fuseki import make_sparql_store
except ImportError:  # pragma: no cover - optional at import time.
    make_sparql_store = None


DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


class QueryTimeout(Exception):
    pass


def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return DEFAULT_PREFIX + query


def _strip_comments(query: str) -> str:
    cleaned_lines = []
    for line in query.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


def _canonical_query_key(query: str) -> str:
    return " ".join(_ensure_prefixes(_strip_comments(query)).split()).lower()


def _normalize_term(term: Any) -> str:
    if term is None:
        return "NULL"
    try:
        return term.n3()
    except Exception:
        return str(term)


@contextmanager
def _time_limit(seconds: Optional[float]):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise QueryTimeout(f"Query exceeded {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _json_literal_n3(binding: Dict[str, Any]) -> str:
    value = str(binding.get("value", ""))
    quoted = json.dumps(value, ensure_ascii=False)
    datatype = str(binding.get("datatype", "")).strip()
    lang = str(binding.get("xml:lang", "")).strip()
    if lang:
        return f"{quoted}@{lang}"
    if datatype and datatype != "http://www.w3.org/2001/XMLSchema#string":
        return f"{quoted}^^<{datatype}>"
    return quoted


def _json_binding_n3(binding: Optional[Dict[str, Any]]) -> str:
    if not isinstance(binding, dict):
        return "NULL"
    kind = str(binding.get("type", "")).strip().lower()
    value = str(binding.get("value", ""))
    if kind == "uri":
        return f"<{value}>"
    if kind in {"bnode", "blank"}:
        return f"_:{value}"
    if kind == "literal" or "datatype" in binding or "xml:lang" in binding:
        return _json_literal_n3(binding)
    return json.dumps(value, ensure_ascii=False)


def _run_fuseki_query(
    query: str, fuseki_query_url: str, timeout_s: Optional[float]
) -> Tuple[List[str], List[Tuple[str, ...]]]:
    data = urllib_parse.urlencode({"query": query}).encode("utf-8")
    req = urllib_request.Request(
        fuseki_query_url,
        data=data,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        },
        method="POST",
    )
    timeout = float(timeout_s) if timeout_s and timeout_s > 0 else None
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (socket.timeout, TimeoutError, urllib_error.URLError) as exc:
        reason = getattr(exc, "reason", None)
        if isinstance(exc, (socket.timeout, TimeoutError)) or isinstance(reason, (socket.timeout, TimeoutError)):
            raise QueryTimeout(f"Query exceeded {timeout_s} seconds") from exc
        raise

    if "boolean" in payload:
        return ["ask"], [(str(bool(payload["boolean"])),)]

    head_vars = sorted(str(v) for v in payload.get("head", {}).get("vars", []))
    bindings = payload.get("results", {}).get("bindings", [])
    rows = [tuple(_json_binding_n3(row.get(var)) for var in head_vars) for row in bindings]
    return head_vars, rows


def _run_local_query(graph: "Graph", query: str, timeout_s: Optional[float]) -> Tuple[List[str], List[Tuple[str, ...]]]:
    with _time_limit(timeout_s):
        results = graph.query(query)
        if getattr(results, "type", "") == "ASK":
            return ["ask"], [(str(bool(results.askAnswer)),)]
        selected_vars: Optional[List[str]] = None
        rows: List[Tuple[str, ...]] = []
        for row in results:
            if hasattr(row, "asdict"):
                row_dict = row.asdict()
                if selected_vars is None:
                    selected_vars = sorted(str(k) for k in row_dict.keys())
                rows.append(tuple(_normalize_term(row_dict.get(k)) for k in selected_vars))
            else:
                if selected_vars is None:
                    selected_vars = [f"col{i + 1}" for i in range(len(row))]
                rows.append(tuple(_normalize_term(v) for v in row))
        return selected_vars or [], rows


def _execution_graph(graph_path: str) -> "Graph":
    if Graph is None:
        raise RuntimeError("rdflib is required to execute candidates locally.")
    graph = Graph()
    graph.parse(graph_path, format="turtle")
    return graph


class Executor:
    """Executes candidate queries with a short timeout and a de-dup cache."""

    def __init__(self, graph_path: str, fuseki_query_url: str, timeout_s: float) -> None:
        self.fuseki_query_url = fuseki_query_url.strip()
        self.timeout_s = timeout_s
        self.graph = None
        if not self.fuseki_query_url:
            self.graph = _execution_graph(graph_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.executions = 0
        self.cache_hits = 0

    def run(self, query: str) -> Dict[str, Any]:
        key = _canonical_query_key(query)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        full_query = _ensure_prefixes(_strip_comments(query))
        self.executions += 1
        try:
            if self.fuseki_query_url:
                selected_vars, rows = _run_fuseki_query(full_query, self.fuseki_query_url, self.timeout_s)
            else:
                selected_vars, rows = _run_local_query(self.graph, full_query, self.timeout_s)
            outcome: Dict[str, Any] = {
                "status": "ok",
                "selected_vars": selected_vars,
                "row_count": len(rows),
                "rows": rows,
            }
        except QueryTimeout:
            outcome = {"status": "timeout"}
        except Exception as exc:  # pragma: no cover - depends on external graph/Fuseki.
            outcome = {"status": "error", "error": str(exc)[:200]}

        self._cache[key] = outcome
        return outcome


def _execution_signature(
    outcome: Dict[str, Any],
    preview_rows: int,
    *,
    candidate_key: int = 0,
    strict_nonempty: bool = False,
) -> str:
    """Build a signature clustering candidates by execution outcome.

    ``strict_nonempty`` controls whether timeouts, errors, and successful
    but empty (0-row) results are allowed to cluster with other candidates
    that failed/returned nothing for a different reason. When False (the
    default, "lenient" variant), all timeouts collapse into one bucket and
    all 0-row successes with the same projected variables collapse into
    one bucket, which is appropriate for measuring "does the user see a
    distinguishable outcome" but conflates genuinely equivalent answers
    with merely-coincidental shared failure/emptiness. When True (the
    "strict" variant), every timeout, every error, and every 0-row success
    gets its own unique signature via ``candidate_key``, so only candidates
    that produced the *same non-empty* result are ever clustered together.
    """
    if outcome["status"] == "timeout":
        return f"timeout:{candidate_key}" if strict_nonempty else "timeout"
    if outcome["status"] == "error":
        return f"error:{candidate_key}:{outcome['error']}" if strict_nonempty else f"error:{outcome['error']}"
    if strict_nonempty and outcome["row_count"] == 0:
        return f"empty:{candidate_key}"
    preview = sorted(outcome["rows"])[: max(0, preview_rows)]
    payload = {
        "vars": outcome["selected_vars"],
        "row_count": outcome["row_count"],
        "preview": preview,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return "sig:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _details(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("details") or payload.get("rows") or []
    return [row for row in rows if isinstance(row, dict)]


def _candidate_score(candidate: Dict[str, Any], score_key: str, fallback_rank: int) -> float:
    for key in (score_key, "ml_score", "selection_score", "score"):
        value = candidate.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return -float(fallback_rank)


def _softmax(scores: List[float], temperature: float) -> List[float]:
    if not scores:
        return []
    temp = max(float(temperature), 1e-9)
    top = max(scores)
    shifted = [(s - top) / temp for s in scores]
    exps = [math.exp(max(min(x, 60), -60)) for x in shifted]
    total = sum(exps)
    return [x / total for x in exps] if total else [1.0 / len(scores)] * len(scores)


def _shannon_entropy_nats(probs: Sequence[float]) -> float:
    nonzero = [p for p in probs if p > 0]
    if len(nonzero) <= 1:
        return 0.0
    return -sum(p * math.log(p) for p in nonzero)


def _normalized_entropy(h: float, group_count: int) -> float:
    if group_count <= 1:
        return 0.0
    return h / math.log(group_count)


def analyze(
    results_path: str,
    *,
    graph_path: str,
    fuseki_query_url: str,
    score_key: str,
    temperature: float,
    timeout_s: float,
    preview_rows: int,
    min_candidates: int,
    limit: int,
) -> Dict[str, Any]:
    payload = _load_json(results_path)
    details = _details(payload)
    if limit > 0:
        details = details[:limit]

    executor = Executor(graph_path=graph_path, fuseki_query_url=fuseki_query_url, timeout_s=timeout_s)

    rows: List[Dict[str, Any]] = []
    h_raw_values: List[float] = []
    h_sig_values: List[float] = []
    h_sig_strict_values: List[float] = []

    for detail in details:
        candidates = detail.get("candidates") or []
        if not isinstance(candidates, list) or len(candidates) < max(2, min_candidates):
            continue

        scores = [_candidate_score(c, score_key, idx) for idx, c in enumerate(candidates)]
        probs = _softmax(scores, temperature)
        h_raw = _shannon_entropy_nats(probs)
        h_raw_norm = _normalized_entropy(h_raw, len(probs))

        grouped: Dict[str, float] = {}
        grouped_strict: Dict[str, float] = {}
        group_status: Dict[str, Dict[str, Any]] = {}
        group_status_strict: Dict[str, Dict[str, Any]] = {}
        error_count = 0
        timeout_count = 0
        nonempty_count = 0
        for idx, (prob, candidate) in enumerate(zip(probs, candidates)):
            query = str(candidate.get("query") or "")
            outcome = executor.run(query) if query.strip() else {"status": "error", "error": "empty_query"}
            if outcome["status"] == "error":
                error_count += 1
            elif outcome["status"] == "timeout":
                timeout_count += 1
            elif outcome.get("row_count", 0) > 0:
                nonempty_count += 1

            sig = _execution_signature(outcome, preview_rows, candidate_key=idx, strict_nonempty=False)
            grouped[sig] = grouped.get(sig, 0.0) + prob
            meta = group_status.setdefault(
                sig,
                {"status": outcome["status"], "row_count": outcome.get("row_count"), "candidate_count": 0},
            )
            meta["candidate_count"] += 1

            sig_strict = _execution_signature(outcome, preview_rows, candidate_key=idx, strict_nonempty=True)
            grouped_strict[sig_strict] = grouped_strict.get(sig_strict, 0.0) + prob
            meta_strict = group_status_strict.setdefault(
                sig_strict,
                {"status": outcome["status"], "row_count": outcome.get("row_count"), "candidate_count": 0},
            )
            meta_strict["candidate_count"] += 1

        group_probs = list(grouped.values())
        h_sig = _shannon_entropy_nats(group_probs)
        h_sig_norm = _normalized_entropy(h_sig, len(group_probs))

        group_probs_strict = list(grouped_strict.values())
        h_sig_strict = _shannon_entropy_nats(group_probs_strict)
        h_sig_strict_norm = _normalized_entropy(h_sig_strict, len(group_probs_strict))

        h_raw_values.append(h_raw)
        h_sig_values.append(h_sig)
        h_sig_strict_values.append(h_sig_strict)

        rows.append(
            {
                "id": detail.get("id") or detail.get("question_id") or detail.get("request_id"),
                "question": detail.get("question") or detail.get("effective_question"),
                "candidate_count": len(candidates),
                "nonempty_candidates": nonempty_count,
                "failed_or_empty_candidates": len(candidates) - nonempty_count,
                "signature_count": len(grouped),
                "signature_count_strict": len(grouped_strict),
                "execution_errors": error_count,
                "execution_timeouts": timeout_count,
                "H_raw": h_raw,
                "H_raw_norm": h_raw_norm,
                "H_sig": h_sig,
                "H_sig_norm": h_sig_norm,
                "H_sig_strict": h_sig_strict,
                "H_sig_strict_norm": h_sig_strict_norm,
                "entropy_reduction": h_raw - h_sig,
                "entropy_reduction_norm": h_raw_norm - h_sig_norm,
                "entropy_reduction_strict": h_raw - h_sig_strict,
                "entropy_reduction_strict_norm": h_raw_norm - h_sig_strict_norm,
                "groups": [
                    {"signature": sig, "probability": grouped[sig], **group_status[sig]}
                    for sig in sorted(grouped, key=lambda s: -grouped[s])
                ],
                "groups_strict": [
                    {"signature": sig, "probability": grouped_strict[sig], **group_status_strict[sig]}
                    for sig in sorted(grouped_strict, key=lambda s: -grouped_strict[s])
                ],
            }
        )

    rows.sort(key=lambda row: (-float(row["entropy_reduction"]), str(row["id"])))
    total = len(rows)
    avg_h_raw = sum(h_raw_values) / total if total else 0.0
    avg_h_sig = sum(h_sig_values) / total if total else 0.0
    avg_h_sig_strict = sum(h_sig_strict_values) / total if total else 0.0

    return {
        "results": results_path,
        "graph_path": graph_path if not fuseki_query_url else "",
        "fuseki_query_url": fuseki_query_url,
        "score_key": score_key,
        "temperature": temperature,
        "timeout_s": timeout_s,
        "preview_rows": preview_rows,
        "summary": {
            "questions": total,
            "avg_H_raw": avg_h_raw,
            "avg_H_sig": avg_h_sig,
            "avg_H_sig_strict": avg_h_sig_strict,
            "avg_entropy_reduction": avg_h_raw - avg_h_sig,
            "avg_entropy_reduction_strict": avg_h_raw - avg_h_sig_strict,
            "candidate_executions": executor.executions,
            "candidate_cache_hits": executor.cache_hits,
        },
        "rows": rows,
        "interpretation": (
            "Two clustering variants are reported. H_sig (lenient) clusters candidates by "
            "execution outcome including shared failure modes: all timeouts for a question "
            "collapse into one bucket, and all successful-but-empty (0-row) results with the "
            "same projected variables collapse into one bucket, alongside genuine non-empty "
            "result equivalence. This answers 'does the user see a distinguishable outcome' but "
            "can understate ambiguity between candidates that merely failed/matched-nothing for "
            "different underlying reasons. H_sig_strict never clusters timeouts, errors, or "
            "0-row results with one another -- each gets a unique signature -- so it only "
            "collapses candidates that produced the exact same non-empty result. The gap between "
            "H_raw and H_sig_strict is therefore a conservative, reviewer-safe lower bound on how "
            "much raw candidate-level entropy is attributable to genuine execution-level answer "
            "equivalence, uncontaminated by shared failure or shared emptiness."
        ),
    }


def _write_csv(report: Dict[str, Any], out_csv: str) -> None:
    fieldnames = [
        "id",
        "question",
        "candidate_count",
        "nonempty_candidates",
        "failed_or_empty_candidates",
        "signature_count",
        "signature_count_strict",
        "execution_errors",
        "execution_timeouts",
        "H_raw",
        "H_raw_norm",
        "H_sig",
        "H_sig_norm",
        "H_sig_strict",
        "H_sig_strict_norm",
        "entropy_reduction",
        "entropy_reduction_norm",
        "entropy_reduction_strict",
        "entropy_reduction_strict_norm",
    ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute every candidate SPARQL query, cluster by execution signature "
            "(selected variables + row count + normalized preview rows), and compute "
            "H_sig / H_sig_norm entropy over the resulting groups."
        )
    )
    parser.add_argument("--results", required=True, help="Selection/analysis JSON with details[].candidates[].query")
    parser.add_argument("--graph", default="data/infineon/graph.ttl", help="Local Turtle graph (ignored if --fuseki-query-url is set)")
    parser.add_argument(
        "--fuseki-query-url",
        default=os.environ.get("FUSEKI_QUERY_URL", ""),
        help="Fuseki SPARQL query endpoint; defaults to $FUSEKI_QUERY_URL, else falls back to --graph",
    )
    parser.add_argument("--score-key", default="ml_score", help="Candidate score field used for the softmax probabilities")
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--timeout-s", type=float, default=5.0, help="Per-candidate execution timeout in seconds")
    parser.add_argument("--preview-rows", type=int, default=5, help="Rows retained (after sorting) in each execution signature")
    parser.add_argument("--min-candidates", type=int, default=2, help="Skip questions with fewer candidates than this")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of questions processed (0 = no cap)")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    report = analyze(
        args.results,
        graph_path=args.graph,
        fuseki_query_url=args.fuseki_query_url,
        score_key=args.score_key,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
        preview_rows=args.preview_rows,
        min_candidates=args.min_candidates,
        limit=args.limit,
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_csv(report, args.out_csv)

    print("===== EXECUTION-SIGNATURE ENTROPY (H_sig) =====")
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON: {args.out_json}")
    print(f"CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()
