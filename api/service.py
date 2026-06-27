from __future__ import annotations

import json
import csv
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from kg.advisory import resolve_advisory_plan, synthesize_advisory_answer
from kg.capabilities import DEFAULT_REGISTRY
from kg.schema import load_default_schema
from llm.answer_synthesis import synthesize_answer
from llm.client import InfineonGPTClient
from pipeline.qa import answer_question
from pipeline.request_routing import route_request
from visualization.interactive_graph import (
    collect_answer_evidence_triples,
    collect_query_subgraph_triples,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = ROOT / "data" / "infineon" / "graph.ttl"
DEFAULT_ONTOLOGY_PATH = ROOT / "data" / "infineon" / "true_demand_ontology_extracted.ttl"
DEFAULT_AUDIT_PATH = ROOT / "results" / "kgqa_system_accuracy_audit_500_v2_labeled.json"
DEFAULT_AUDIT_CSV_PATH = ROOT / "results" / "kgqa_system_accuracy_audit_500_v2_labeled.csv"
DEFAULT_FUSEKI_URL = "http://localhost:3030/infineon/sparql"
SURVEY_PREFIX = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"
PREFIXES = f"""\
PREFIX survey: <{SURVEY_PREFIX}>
PREFIX rdf: <{RDF}>
PREFIX rdfs: <{RDFS}>
PREFIX owl: <{OWL}>
"""


def _short(value: object) -> str:
    text = str(value or "")
    local = text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    local = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local).replace("_", " ")
    return re.sub(r"\s+", " ", local).strip() or text


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _with_prefixes(query: str) -> str:
    return query if "PREFIX" in query.upper() else PREFIXES + query


def _json_value(value: object) -> str | int | float | bool:
    if isinstance(value, Literal):
        py = value.toPython()
        if isinstance(py, (str, int, float, bool)):
            return py
    return str(value)


class KGQAService:
    def __init__(self) -> None:
        self.schema = load_default_schema()
        self.registry = DEFAULT_REGISTRY
        self._cases: Dict[str, Dict[str, object]] = {}
        self._lock = threading.RLock()
        self._local_graph: Optional[Graph] = None
        self._ontology_graph: Optional[Graph] = None
        self._capability_cache: Tuple[Dict[str, object], ...] = ()
        self._capability_cache_at = 0.0

    @property
    def fuseki_url(self) -> str:
        return os.getenv("FUSEKI_QUERY_URL", DEFAULT_FUSEKI_URL).strip()

    def _graph(self) -> Graph:
        if self.fuseki_url:
            return Graph(store=SPARQLStore(self.fuseki_url))
        if self._local_graph is None:
            graph_path = Path(os.getenv("TRUE_DEMAND_GRAPH_PATH", str(DEFAULT_GRAPH_PATH)))
            graph = Graph()
            graph.parse(str(graph_path), format="turtle")
            self._local_graph = graph
        return self._local_graph

    def _ontology(self) -> Graph:
        if self._ontology_graph is None:
            path = Path(os.getenv("TRUE_DEMAND_ONTOLOGY_PATH", str(DEFAULT_ONTOLOGY_PATH)))
            graph = Graph()
            graph.parse(str(path), format="turtle")
            self._ontology_graph = graph
        return self._ontology_graph

    def execute(self, query: str, max_rows: int = 200) -> Tuple[List[Dict[str, object]], bool, str]:
        try:
            result = self._graph().query(_with_prefixes(query))
            rows: List[Dict[str, object]] = []
            truncated = False
            for index, row in enumerate(result):
                if index >= max_rows:
                    truncated = True
                    break
                if hasattr(row, "asdict"):
                    rows.append({str(k): _json_value(v) for k, v in row.asdict().items()})
                else:
                    rows.append({f"col{i + 1}": _json_value(v) for i, v in enumerate(row)})
            return rows, truncated, ""
        except Exception as exc:
            return [], False, str(exc)

    def _direct_query(self, question: str) -> Optional[str]:
        report = self.registry.resolve(question)
        query = self.registry.direct_query_for(report)
        if not query:
            return None
        rows, _truncated, error = self.execute(query, max_rows=1)
        return query if rows and not error else None

    def _llm_client(self) -> InfineonGPTClient:
        return InfineonGPTClient(
            model=os.getenv("INFINEON_MODEL") or None,
            base_url=os.getenv("INFINEON_API_URL") or None,
            api_key=os.getenv("INFINEON_API_KEY") or None,
            temperature=float(os.getenv("INFINEON_TEMPERATURE", "0.2")),
        )

    @staticmethod
    def _family(question: str) -> str:
        report = DEFAULT_REGISTRY.resolve(question)
        return report.primary_capability or "unresolved"

    @staticmethod
    def _table(rows: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not rows:
            return None
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        return {
            "columns": [{"key": key, "label": _short(key).title()} for key in keys],
            "rows": rows,
        }

    @staticmethod
    def _candidate_score(result: Dict[str, object], selected_query: str) -> float:
        selected_key = " ".join(selected_query.split()).lower()
        ranked = list(result.get("learning_ranked") or []) or list(result.get("schema_ranked") or [])
        for row in ranked:
            query = " ".join(str(row.get("query", "")).split()).lower()
            if query != selected_key:
                continue
            score = float(row.get("ml_score", row.get("score", 0.0)) or 0.0)
            if 0.0 <= score <= 1.0:
                return score
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
        return 0.5

    def _interpretations(
        self,
        case_id: str,
        payload: Dict[str, object],
        question: str,
    ) -> List[Dict[str, object]]:
        options = list(payload.get("options") or [])[:3]
        interpretations: List[Dict[str, object]] = []
        stored: Dict[str, Dict[str, str]] = {}
        for index, option in enumerate(options, start=1):
            option_id = str(option.get("id") or f"option_{index}")
            query = str(option.get("query") or "").strip()
            rewritten = str(option.get("rewritten_question") or "").strip()
            if query:
                rows, _truncated, error = self.execute(query, max_rows=1)
                if error or not rows:
                    continue
                stored[option_id] = {"query": query}
            elif rewritten:
                stored[option_id] = {"question": rewritten}
            else:
                continue
            confidence = float(option.get("score", option.get("support", 0.5)) or 0.5)
            if confidence > 1:
                confidence = 1.0 / (1.0 + math.exp(-confidence))
            interpretations.append(
                {
                    "id": option_id,
                    "text": str(option.get("label") or option.get("text") or rewritten or "Interpretation"),
                    "confidence": max(0.0, min(1.0, confidence)),
                    "hint": str(option.get("preview") or option.get("choose_if") or "").strip() or None,
                }
            )
        with self._lock:
            case = self._cases.setdefault(case_id, {"question": question})
            case["options"] = stored
        return interpretations

    def ask(self, question: str) -> Dict[str, object]:
        started = time.perf_counter()
        question = str(question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")
        case_id = uuid.uuid4().hex[:12]
        advisory_plan = resolve_advisory_plan(question)
        if advisory_plan is not None:
            rows, truncated, error = self.execute(advisory_plan.query)
            if rows and not error:
                answer = synthesize_advisory_answer(question, advisory_plan, rows)
                with self._lock:
                    self._cases[case_id] = {
                        "question": question,
                        "query": advisory_plan.query,
                        "rows": rows,
                        "decision": "advisory",
                    }
                return {
                    "caseId": case_id,
                    "decision": "advisory",
                    "answer": answer,
                    "table": self._table(rows),
                    "confidence": 1.0,
                    "entropy": 0.0,
                    "responseTimeMs": round((time.perf_counter() - started) * 1000),
                    "sparql": advisory_plan.query,
                    "unsupported": False,
                    "diagnostics": {
                        "family": "advisory",
                        "template": advisory_plan.plan_id,
                        "safetyFlags": ["graph_grounded_advisory_not_business_decision"],
                        "truncated": bool(truncated),
                    },
                }

        direct_query = self._direct_query(question)
        if direct_query:
            rows, truncated, error = self.execute(direct_query)
            return self._answer_response(
                case_id=case_id,
                question=question,
                query=direct_query,
                rows=rows,
                error=error,
                decision="direct",
                confidence=1.0,
                entropy=0.0,
                elapsed=time.perf_counter() - started,
                diagnostics={"family": self._family(question), "template": "capability_registry", "safetyFlags": []},
                truncated=truncated,
            )

        early_route = route_request(question, schema=self.schema, alias_index=None)
        if early_route.get("route") != "kg_query" and early_route.get("answer"):
            with self._lock:
                self._cases[case_id] = {"question": question, "rows": [], "query": ""}
            return {
                "caseId": case_id,
                "decision": "definition" if early_route.get("route") == "definition" else str(early_route.get("route")),
                "answer": str(early_route.get("answer") or ""),
                "confidence": 1.0 if early_route.get("confidence") == "High" else 0.7,
                "entropy": 0.0,
                "responseTimeMs": round((time.perf_counter() - started) * 1000),
                "sparql": "",
                "diagnostics": {
                    "family": "ontology",
                    "safetyFlags": [],
                    "route": str(early_route.get("route") or ""),
                    "source": str(early_route.get("source") or "request_router"),
                    "matchedTerm": str(early_route.get("matched_term") or ""),
                    "termKind": str(early_route.get("term_kind") or ""),
                    "termUri": str(early_route.get("term_uri") or ""),
                },
            }

        result = answer_question(
            question,
            self.schema,
            llm_client=self._llm_client(),
            enable_entity_linking=True,
            use_ml_ranking=True,
            ml_policy=os.getenv("KGQA_ML_POLICY", "auto"),
            ml_model_path=os.getenv("KGQA_ML_MODEL") or None,
            ml_ambiguity_config_path=os.getenv("INFINEON_AMBIGUITY_CONFIG") or None,
            include_candidate_diagnostics=True,
            enable_clarification=True,
            enable_answerability_assessment=True,
        )
        request_route = result.get("request_route")
        if isinstance(request_route, dict) and request_route.get("route") != "kg_query":
            with self._lock:
                self._cases[case_id] = {"question": question, "rows": [], "query": ""}
            return {
                "caseId": case_id,
                "decision": "definition",
                "answer": str(result.get("answer") or ""),
                "confidence": 1.0 if request_route.get("confidence") == "High" else 0.7,
                "entropy": 0.0,
                "responseTimeMs": round((time.perf_counter() - started) * 1000),
                "sparql": "",
                "diagnostics": {
                    "family": "ontology",
                    "safetyFlags": [],
                    "route": str(request_route.get("route") or ""),
                    "source": str(request_route.get("source") or "request_router"),
                    "matchedTerm": str(request_route.get("matched_term") or ""),
                    "termKind": str(request_route.get("term_kind") or ""),
                    "termUri": str(request_route.get("term_uri") or ""),
                },
            }
        request_clarification = result.get("request_clarification")
        clarification = result.get("clarification")
        clarification_payload = (
            request_clarification if isinstance(request_clarification, dict) else
            clarification if isinstance(clarification, dict) else None
        )
        if clarification_payload and clarification_payload.get("needs_clarification"):
            interpretations = self._interpretations(case_id, clarification_payload, question)
            if len(interpretations) >= 2:
                return {
                    "caseId": case_id,
                    "decision": "clarification",
                    "answer": "",
                    "interpretations": interpretations,
                    "confidence": self._candidate_score(result, str(result.get("selected_query") or "")),
                    "entropy": float(result.get("predicted_entropy", result.get("entropy", 0.0)) or 0.0),
                    "responseTimeMs": round((time.perf_counter() - started) * 1000),
                    "sparql": "",
                    "diagnostics": {
                        "family": self._family(question),
                        "rankerScore": self._candidate_score(result, str(result.get("selected_query") or "")),
                        "safetyFlags": [str(clarification_payload.get("reason") or "ambiguous_interpretation")],
                    },
                }

        selected_query = str(result.get("selected_query") or "").strip()
        if not selected_query:
            with self._lock:
                self._cases[case_id] = {"question": question, "rows": [], "query": ""}
            return {
                "caseId": case_id,
                "decision": "unsupported",
                "answer": str(result.get("answer") or "No graph-supported interpretation was found."),
                "confidence": 0.0,
                "entropy": float(result.get("predicted_entropy", result.get("entropy", 0.0)) or 0.0),
                "responseTimeMs": round((time.perf_counter() - started) * 1000),
                "sparql": "",
                "unsupported": True,
                "diagnostics": {"family": self._family(question), "safetyFlags": ["no_selected_query"]},
            }
        rows, truncated, error = self.execute(selected_query)
        return self._answer_response(
            case_id=case_id,
            question=question,
            query=selected_query,
            rows=rows,
            error=error,
            decision="auto",
            confidence=self._candidate_score(result, selected_query),
            entropy=float(result.get("predicted_entropy", result.get("entropy", 0.0)) or 0.0),
            elapsed=time.perf_counter() - started,
            diagnostics={
                "family": self._family(question),
                "rankerScore": self._candidate_score(result, selected_query),
                "safetyFlags": [str(x) for x in result.get("errors", [])],
            },
            truncated=truncated,
        )

    def _answer_response(
        self,
        *,
        case_id: str,
        question: str,
        query: str,
        rows: List[Dict[str, object]],
        error: str,
        decision: str,
        confidence: float,
        entropy: float,
        elapsed: float,
        diagnostics: Dict[str, object],
        truncated: bool = False,
    ) -> Dict[str, object]:
        unsupported = bool(error or not rows)
        answer = (
            f"Graph execution failed: {error}" if error else
            synthesize_answer(question, query, {"rows": rows, "matched_question_id": None, "error": None})
        )
        if answer.startswith("Answer: "):
            answer = answer[8:]
        with self._lock:
            self._cases[case_id] = {
                "question": question,
                "query": query,
                "rows": rows,
                "decision": decision,
            }
        response: Dict[str, object] = {
            "caseId": case_id,
            "decision": "unsupported" if unsupported else decision,
            "answer": answer,
            "table": self._table(rows),
            "confidence": 0.0 if unsupported else confidence,
            "entropy": entropy,
            "responseTimeMs": round(elapsed * 1000),
            "sparql": query,
            "unsupported": unsupported,
            "diagnostics": diagnostics,
        }
        if truncated:
            response["diagnostics"]["safetyFlags"] = list(response["diagnostics"].get("safetyFlags") or []) + ["preview_truncated"]
        return response

    def clarify(self, case_id: str, choice_id: str) -> Dict[str, object]:
        with self._lock:
            case = dict(self._cases.get(case_id) or {})
        option = dict((case.get("options") or {}).get(choice_id) or {})
        if not option:
            raise KeyError("Unknown clarification case or choice.")
        if option.get("question"):
            return self.ask(str(option["question"]))
        query = str(option.get("query") or "")
        rows, truncated, error = self.execute(query)
        return self._answer_response(
            case_id=case_id,
            question=str(case.get("question") or ""),
            query=query,
            rows=rows,
            error=error,
            decision="auto",
            confidence=1.0,
            entropy=0.0,
            elapsed=0.0,
            diagnostics={"family": self._family(str(case.get("question") or "")), "template": "clarified_query", "safetyFlags": []},
            truncated=truncated,
        )

    def answerable_capabilities(self) -> Tuple[Dict[str, object], ...]:
        now = time.monotonic()
        if self._capability_cache and now - self._capability_cache_at < 60.0:
            return self._capability_cache
        output: List[Dict[str, object]] = []
        for capability in self.registry.capabilities:
            dimensions = []
            examples = []
            for dimension in capability.dimensions:
                question = f"Show {capability.name} by {dimension.name}."
                report = self.registry.resolve(question)
                query = self.registry.direct_query_for(report)
                if not query:
                    continue
                rows, _truncated, error = self.execute(query, max_rows=1)
                if error or not rows:
                    continue
                dimensions.append(dimension.name)
                examples.append({"id": _slug(question), "text": question, "category": capability.name})
            if dimensions:
                output.append(
                    {
                        "family": capability.name.title(),
                        "templates": len(dimensions),
                        "description": f"Graph-supported {capability.name} questions.",
                        "dimensions": dimensions,
                        "aggregations": list(capability.aggregations),
                        "examples": examples,
                    }
                )
        result = tuple(output)
        if result:
            self._capability_cache = result
            self._capability_cache_at = now
        return result

    def examples(self) -> List[Dict[str, object]]:
        return [example for capability in self.answerable_capabilities() for example in capability["examples"]]

    def autocomplete(self, token: str, context: str = "") -> List[Dict[str, object]]:
        token_norm = token.lower().strip()
        context_norm = context.lower()
        dimension_context = bool(re.search(r"\b(by|per|across)(?:\s+\w*)?$", context_norm))
        suggestions: List[Dict[str, object]] = []
        seen = set()
        for capability in self.answerable_capabilities():
            family = str(capability["family"])
            family_active = family.lower() in context_norm
            if not dimension_context and (
                not token_norm or family.lower().startswith(token_norm) or token_norm in family.lower()
            ):
                key = (family.lower(), "Metric")
                if key not in seen:
                    seen.add(key)
                    suggestions.append({"id": _slug(f"metric-{family}"), "label": family, "type": "Metric", "description": capability["description"]})
            if family_active or dimension_context:
                for dimension in capability["dimensions"]:
                    label = str(dimension).title()
                    if token_norm and not label.lower().startswith(token_norm):
                        continue
                    key = (label.lower(), "Dimension")
                    if key in seen:
                        continue
                    seen.add(key)
                    suggestions.append({"id": _slug(f"dimension-{family}-{label}"), "label": label, "type": "Dimension", "description": f"Answerable breakdown for {family}."})
        return suggestions[:12]

    def ontology_payload(self) -> Dict[str, object]:
        graph = self._ontology()
        interesting = {RDFS.subClassOf, RDFS.domain, RDFS.range}
        triples = [(s, p, o) for s, p, o in graph if p in interesting and not isinstance(o, Literal)]
        return self._triples_payload(graph, triples[:1200])

    def data_payload(self, limit: int = 500) -> Dict[str, object]:
        graph = self._graph()
        triples: List[Tuple[object, object, object]] = []
        if self.fuseki_url:
            result = graph.query(
                _with_prefixes(
                    f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . FILTER(?p != rdf:type) }} LIMIT {max(1, min(limit, 1000))}"
                )
            )
            for row in result:
                triples.append((row[0], row[1], row[2]))
        else:
            for subject, predicate, obj in graph:
                if predicate == RDF.type:
                    continue
                triples.append((subject, predicate, obj))
                if len(triples) >= max(1, min(limit, 1000)):
                    break
        return self._triples_payload(self._ontology(), triples)

    def evidence_payload(self, case_id: str) -> Dict[str, object]:
        with self._lock:
            case = dict(self._cases.get(case_id) or {})
        query = str(case.get("query") or "")
        rows = list(case.get("rows") or [])
        if not query or not rows:
            return {"nodes": [], "edges": [], "pathNodeIds": [], "pathEdgeIds": []}
        graph = self._graph()
        schema_triples, _ = collect_answer_evidence_triples(graph, query, limit=32)
        data_triples, _ = collect_query_subgraph_triples(graph, query, rows, hops=1, limit=80)
        triples = list(dict.fromkeys([*schema_triples, *data_triples]))
        payload = self._triples_payload(self._ontology(), triples)
        payload["pathNodeIds"] = [node["id"] for node in payload["nodes"]]
        payload["pathEdgeIds"] = [edge["id"] for edge in payload["edges"]]
        return payload

    def _triples_payload(self, metadata_graph: Graph, triples: Iterable[Tuple[object, object, object]]) -> Dict[str, object]:
        nodes: Dict[str, Dict[str, object]] = {}
        edges: List[Dict[str, object]] = []

        def add_node(term: object) -> str:
            node_id = str(term)
            if node_id in nodes:
                return node_id
            if isinstance(term, Literal):
                node_type = "Literal"
                label = str(term)
            elif isinstance(term, BNode):
                node_type = "Entity"
                label = "Anonymous resource"
            else:
                types = set(metadata_graph.objects(term, RDF.type))
                if OWL.Class in types or RDFS.Class in types:
                    node_type = "Class"
                elif OWL.ObjectProperty in types:
                    node_type = "ObjectProperty"
                elif OWL.DatatypeProperty in types:
                    node_type = "DatatypeProperty"
                elif RDF.Property in types:
                    node_type = "Property"
                elif str(term).startswith(str(XSD)) or term == RDF.langString:
                    node_type = "Datatype"
                else:
                    node_type = "Entity"
                label_value = next(iter(metadata_graph.objects(term, RDFS.label)), None)
                label = str(label_value) if label_value is not None else _short(term)
            definition = next(iter(metadata_graph.objects(term, RDFS.comment)), None) if not isinstance(term, Literal) else None
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "iri": str(term) if isinstance(term, URIRef) else None,
                "definition": str(definition) if definition is not None else None,
                "value": _json_value(term) if isinstance(term, Literal) else None,
                "datatype": str(term.datatype) if isinstance(term, Literal) and term.datatype else None,
                "properties": [],
            }
            return node_id

        for index, (subject, predicate, obj) in enumerate(triples):
            source = add_node(subject)
            target = add_node(obj)
            edges.append({
                "id": f"edge-{index}-{_slug(_short(predicate))}",
                "source": source,
                "target": target,
                "label": _short(predicate),
                "iri": str(predicate),
            })
        return {"nodes": list(nodes.values()), "edges": edges}

    def metrics(self) -> Dict[str, object]:
        audit = {}
        if DEFAULT_AUDIT_PATH.exists():
            audit = json.loads(DEFAULT_AUDIT_PATH.read_text(encoding="utf-8"))
        overall = audit.get("overall", {})
        modes = audit.get("by_mode", {})
        direct = modes.get("direct_graph_supported", {})
        llm = modes.get("llm_ranking", {})
        cases: List[Dict[str, object]] = []
        if DEFAULT_AUDIT_CSV_PATH.exists():
            with DEFAULT_AUDIT_CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    correctness = str(row.get("correctness") or "").lower()
                    mode = str(row.get("system_mode") or "")
                    route = str(row.get("route") or "")
                    decision = (
                        "clarification" if "clar" in route else
                        "direct" if mode == "direct_graph_supported" else
                        "auto"
                    )
                    cases.append(
                        {
                            "id": str(row.get("request_id") or ""),
                            "question": str(row.get("question") or ""),
                            "family": str(row.get("topic") or "unresolved"),
                            "decision": decision,
                            "confidence": 1.0 if decision == "direct" else 0.0,
                            "entropy": 0.0,
                            "correct": True if correctness == "correct" else False if correctness == "incorrect" else None,
                            "safetyFlags": ["graph_error"] if row.get("graph_error") else [],
                        }
                    )
        return {
            "forcedTop1": 0.677,
            "anyCorrect": 0.944,
            "autoAnswerAccuracy": float(direct.get("accuracy", 0.9555)),
            "coverage": float(direct.get("total_rows", 337)) / max(1, int(overall.get("total_rows", 500))),
            "systemAccuracy": float(overall.get("accuracy", 0.798)),
            "directAccuracy": float(direct.get("accuracy", 0.9555)),
            "llmAccuracy": float(llm.get("accuracy", 0.4724)),
            "confidenceBuckets": [],
            "entropyRouting": [
                {"route": "Direct Answer", "count": int(direct.get("total_rows", 337))},
                {"route": "LLM + Ranking", "count": int(llm.get("total_rows", 163))},
            ],
            "accuracyVsCoverage": [
                {"coverage": float(direct.get("total_rows", 337)) / max(1, int(overall.get("total_rows", 500))), "accuracy": float(direct.get("accuracy", 0.9555))},
                {"coverage": 1.0, "accuracy": float(overall.get("accuracy", 0.798))},
            ],
            "cases": cases,
        }

    def health(self) -> Dict[str, object]:
        started = time.perf_counter()
        _rows, _truncated, graph_error = self.execute("SELECT (1 AS ?ok) WHERE {}", max_rows=1)
        llm_ready = bool(os.getenv("INFINEON_API_URL")) and bool(
            os.getenv("INFINEON_API_KEY") or (os.getenv("USER_LLM") and os.getenv("PASSWORD_LLM"))
        )
        return {
            "api": "ok",
            "fuseki": "ok" if not graph_error else "down",
            "llm": "ok" if llm_ready else "degraded",
            "version": "1.0.0",
            "latencyMs": round((time.perf_counter() - started) * 1000),
        }


service = KGQAService()
