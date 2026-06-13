# candidate_generation.py
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from kg.schema import KGSchema
from kg.entity_linking import EntityAliasIndex, canonicalize_question_with_index, normalize_alias
from kg.entity_profiles import (
    EntityProfile,
    build_profiles_for_iris,
)
from kg.schema_slices import (
    build_schema_slice,
    full_schema_fallback_enabled,
    infer_schema_slice_route,
    schema_slicing_enabled,
)
from llm.prompts import build_candidate_prompt, build_repair_prompt
from llm.client import InfineonGPTClient
from rdflib.plugins.sparql.parser import parseQuery
from validation.schema import validate_query_schema
from validation.semantic import semantic_coverage_report, validate_query_semantic
from validation.syntax import validate_query_syntax
import re


BASE = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = """\
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""
VALIDATED_RETRIEVAL_SOURCES = [
    BASE / "data" / "infineon" / "kgqa_seed_expansion_round1.json",
    BASE / "data" / "infineon" / "infineon_train.json",
    BASE / "data" / "infineon" / "infineon_dev.json",
]
_VALIDATED_QUERY_PAIR_CACHE: Optional[List[Dict[str, str]]] = None
LLM_CACHE_VERSION = "candidate_generation_v1"


def _llm_cache_enabled() -> bool:
    return os.getenv("INFINEON_ENABLE_LLM_CACHE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _llm_cache_dir() -> Path:
    return Path(os.getenv("INFINEON_LLM_CACHE_DIR", str(BASE / ".cache" / "kgqa_llm")))


def _llm_client_signature(client: object) -> Dict[str, object]:
    return {
        "class": client.__class__.__name__,
        "model": getattr(client, "model", None),
        "temperature": getattr(client, "temperature", None),
        "max_tokens": getattr(client, "max_tokens", None),
        "base_url": getattr(client, "base_url", None),
    }


def _llm_cache_key(prompt: str, k: int, client: object, purpose: str) -> str:
    payload = {
        "version": LLM_CACHE_VERSION,
        "purpose": purpose,
        "k": int(k),
        "client": _llm_client_signature(client),
        "prompt": prompt,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_llm_cache(key: str) -> Optional[List[str]]:
    path = _llm_cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("version") != LLM_CACHE_VERSION:
        return None
    values = payload.get("generated")
    if not isinstance(values, list):
        return None
    return [str(value) for value in values]


def _write_llm_cache(key: str, generated: List[str], metadata: Dict[str, object]) -> None:
    try:
        cache_dir = _llm_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(
                {
                    "version": LLM_CACHE_VERSION,
                    "created_at": time.time(),
                    "generated": list(generated),
                    "metadata": metadata,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _generate_with_cache(client: object, prompt: str, k: int, purpose: str) -> tuple[List[str], Dict[str, object]]:
    key = _llm_cache_key(prompt, k, client, purpose)
    metadata = {
        "llm_cache_enabled": _llm_cache_enabled(),
        "llm_cache_key": key,
        "llm_cache_hit": False,
        "llm_cache_purpose": purpose,
    }
    if _llm_cache_enabled():
        cached = _read_llm_cache(key)
        if cached is not None:
            metadata["llm_cache_hit"] = True
            return cached[:k], metadata

    generated = list(client.generate(prompt, k=k))
    if _llm_cache_enabled():
        _write_llm_cache(
            key,
            generated,
            {
                "purpose": purpose,
                "k": int(k),
                "client": _llm_client_signature(client),
            },
        )
    return generated, metadata


def _normalize_candidate_query(text: str) -> str:
    q = (text or "").strip()
    if not q:
        return ""
    q = q.replace("```sparql", "").replace("```sql", "").replace("```", "").strip()
    q = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", q)
    q = q.strip().strip(",")

    # Keep only the SPARQL part if the model prepends explanations.
    sel_idx = q.upper().find("SELECT")
    if sel_idx > 0:
        q = q[sel_idx:].strip()
    return q


def generate_candidate_prompt(
    question: str,
    schema: KGSchema,
    k: int = 5,
    canonical_question: Optional[str] = None,
    entity_mappings: Optional[List[Dict[str, object]]] = None,
    entity_profiles: Optional[List[EntityProfile]] = None,
    predicted_query_plan_labels: Optional[List[str]] = None,
) -> str:
    return build_candidate_prompt(
        question=question,
        schema=schema,
        k=k,
        canonical_question=canonical_question,
        entity_mappings=entity_mappings,
        entity_profiles=entity_profiles,
        predicted_query_plan_labels=predicted_query_plan_labels,
    )


def _default_client():
    import os

    provider = (os.getenv("LLM_PROVIDER") or os.getenv("LLM_BACKEND", "infineon")).strip().lower()
    if provider == "infiineon":
        provider = "infineon"

    if provider == "infineon":
        return InfineonGPTClient()
    raise ValueError(
        f"Unknown/unsupported LLM backend '{provider}'. "
        "Supported backend: infineon."
    )


def _normalize_generated_output(generated: object) -> List[str]:
    if generated is None:
        return []
    if isinstance(generated, str):
        cleaned = generated.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        split_candidates = re.split(r"\n\d+\.\s*", generated)
        if len(split_candidates) <= 1:
            split_candidates = generated.split("\n")
        return [g.strip() for g in split_candidates if g.strip()]
    if isinstance(generated, list):
        return [str(g).strip() for g in generated if str(g).strip()]
    raise ValueError(f"Unexpected LLM output type: {type(generated)}")


def _candidate_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _ensure_prefixes(query: str) -> str:
    q = str(query or "").strip()
    if not q:
        return q
    if re.search(r"^\s*PREFIX\b", q, flags=re.IGNORECASE):
        return q
    return DEFAULT_PREFIX + "\n" + q


def _load_validated_query_pairs() -> List[Dict[str, str]]:
    global _VALIDATED_QUERY_PAIR_CACHE
    if _VALIDATED_QUERY_PAIR_CACHE is not None:
        return list(_VALIDATED_QUERY_PAIR_CACHE)

    pairs: List[Dict[str, str]] = []
    for path in VALIDATED_RETRIEVAL_SOURCES:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        rows = payload if isinstance(payload, list) else payload.get("cases", [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question", "") or "").strip()
            query = str(row.get("query", "") or row.get("gold_query", "") or "").strip()
            if question and query:
                pairs.append(
                    {
                        "question": question,
                        "query": query,
                        "source": str(path.relative_to(BASE)),
                    }
                )

    _VALIDATED_QUERY_PAIR_CACHE = pairs
    return list(pairs)


def _retrieval_tokens(text: str) -> set:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "by",
        "with",
        "me",
        "can",
        "you",
        "show",
        "give",
        "tell",
        "what",
        "which",
        "how",
        "is",
        "are",
        "do",
        "does",
        "our",
        "we",
    }
    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", str(text or "").lower().replace("-", " "))
        if len(tok) > 1 and tok not in stop
    }
    if "tier" in tokens and "1" in tokens:
        tokens.add("tier1")
    return tokens


def _validated_candidate_score(question: str, validated_question: str) -> float:
    q_tokens = _retrieval_tokens(question)
    v_tokens = _retrieval_tokens(validated_question)
    if not q_tokens or not v_tokens:
        return 0.0

    domain = {
        "actual",
        "forecast",
        "vehicle",
        "sales",
        "sold",
        "month",
        "monthly",
        "year",
        "yearly",
        "region",
        "quarter",
        "future",
        "current",
        "demand",
        "inventory",
        "component",
        "trend",
        "participant",
        "participants",
        "semiconductor",
        "technology",
        "category",
        "order",
        "cancellation",
        "response",
        "shortage",
        "company",
        "companies",
        "autonomous",
        "sae",
        "oem",
        "tier1",
    }
    shared = q_tokens & v_tokens
    shared_domain = shared & domain
    if not shared_domain:
        return 0.0

    exclusive_pairs = ({"oem", "tier1"}, {"oem", "semiconductor"}, {"tier1", "semiconductor"})
    for exclusive in exclusive_pairs:
        q_has = q_tokens & exclusive
        v_has = v_tokens & exclusive
        if q_has and v_has and q_has != v_has:
            return 0.0

    overlap = len(shared) / max(1, len(q_tokens))
    jaccard = len(shared) / max(1, len(q_tokens | v_tokens))
    score = overlap + 0.45 * jaccard + 0.1 * len(shared_domain)

    q_norm = " ".join(str(question or "").lower().split())
    v_norm = " ".join(str(validated_question or "").lower().split())
    required_phrase_groups = [
        (("by month", "monthly", "each month", "per month"), ("month", "monthly")),
        (("by year", "yearly", "annual", "annually", "each year", "per year"), ("year", "yearly", "annual")),
        (("by quarter", "each quarter", "per quarter"), ("quarter",)),
        (("by region", "each region", "per region"), ("region",)),
        (("technology category", "technology categories"), ("technology category", "technology categories")),
        (("vehicle type", "vehicle category"), ("vehicle type", "vehicle category")),
        (("sae level",), ("sae level",)),
        (("response type",), ("response type",)),
        (("inventory trend", "by trend"), ("trend", "inventory trend")),
        (("by component", "per component"), ("component",)),
        (("vehicle sales", "vehicles sold", "sales volume"), ("vehicle sales", "vehicles sold", "sales volume", "sold")),
        (("actual", "actuals", "actual data"), ("actual", "actuals", "actual data")),
        (("forecast", "forecasted", "forecast data"), ("forecast", "forecasted", "forecast data")),
        (("future",), ("future",)),
        (("current",), ("current",)),
    ]
    for required_terms, accepted_terms in required_phrase_groups:
        if any(term in q_norm for term in required_terms) and not any(
            term in v_norm for term in accepted_terms
        ):
            score -= 0.85
    if (
        ("which compan" in q_norm or "list compan" in q_norm or "show compan" in q_norm)
        and any(term in v_norm for term in ["how many", "count", "number of"])
    ):
        score -= 1.1
    if ("which compan" in q_norm or "list compan" in q_norm or "show compan" in q_norm):
        if any(term in v_norm for term in ["split", "status", "yes/no", "versus", "group"]):
            score -= 1.0
        if any(term in v_norm for term in ["list compan", "which compan", "reported semiconductor shortage"]):
            score += 0.45
    if any(term in q_norm for term in ["sales", "sold", "vehicle units"]) and any(
        term in v_norm for term in ["autonomous", "sae level", "driving development"]
    ):
        score -= 1.1

    for phrase in (
        "by region",
        "by quarter",
        "by month",
        "by year",
        "technology category",
        "vehicle type",
        "sae level",
        "actual",
        "forecast",
        "shortage",
        "inventory",
        "order cancellation",
    ):
        if phrase in q_norm and phrase in v_norm:
            score += 0.25
    return score


def _retrieve_validated_candidate_queries(question: str, limit: int = 3) -> List[Dict[str, str]]:
    rows: List[tuple] = []
    for pair in _load_validated_query_pairs():
        score = _validated_candidate_score(question, pair["question"])
        if score < 0.72:
            continue
        rows.append((score, pair))

    rows.sort(key=lambda item: item[0], reverse=True)
    candidates: List[Dict[str, str]] = []
    seen = set()
    for score, pair in rows:
        key = _candidate_key(pair["query"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "query": pair["query"],
                "source": "validated_retrieval",
                "validated_question": pair["question"],
                "validated_source": pair["source"],
                "validated_retrieval_score": f"{score:.4f}",
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def _template_candidate_queries(question: str) -> List[str]:
    q = (question or "").lower()
    templates: List[str] = []

    def add(query: str) -> None:
        key = " ".join(query.split()).lower()
        if key not in {" ".join(t.split()).lower() for t in templates}:
            templates.append(query)

    survey_values = (
        "VALUES (?surveyClass ?surveyType) { "
        "(survey:OEM_Survey 'OEM') "
        "(survey:Tier1_Survey 'Tier1') "
        "(survey:Semiconductor_Survey 'Semiconductor') "
        "} "
    )
    asks_vehicle_sales = (
        "vehicle sales" in q
        or "vehicles sold" in q
        or "vehicle unit" in q
        or ("vehicle" in q and any(w in q for w in ["sold", "sales", "units"]))
    )
    asks_actual_sales = asks_vehicle_sales and any(
        w in q for w in ["actual", "actually", "actuals", "transaction", "transactions", "logged"]
    )
    asks_forecast_sales = asks_vehicle_sales and any(
        w in q for w in ["forecast", "forecasted", "projected", "expecting", "expected"]
    )
    asks_monthly_sales = asks_vehicle_sales and any(
        w in q for w in ["month", "monthly", "time period", "period", "each month"]
    )
    asks_yearly_sales = asks_vehicle_sales and any(
        w in q for w in ["year", "yearly", "annual", "annually", "each year", "per year"]
    )
    asks_order_cancellation = (
        "order cancellation" in q
        or "order-cancellation" in q
        or ("order" in q and "cancellation" in q)
    )
    asks_count_or_sum = any(
        w in q
        for w in [
            "how many",
            "count",
            "number",
            "total",
            "sum",
            "summed",
            "combined",
            "participant",
            "participants",
        ]
    )

    if (
        "region" in q
        and "demand" not in q
        and any(w in q for w in ["available", "names", "list", "give me"])
    ):
        add(
            "SELECT ?name WHERE { "
            "?r a survey:Region ; survey:regionName ?name . "
            "} ORDER BY ?name"
        )

    if (
        "demand" not in q
        and ("survey-origin" in q or "survey origin" in q)
    ) and any(
        w in q for w in ["class", "classes", "represented", "available"]
    ):
        add(
            "SELECT DISTINCT ?type WHERE { "
            "?s a ?type . "
            "FILTER(?type IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey)) "
            "} ORDER BY ?type"
        )

    if "tier1" in q and "automotive" in q and "current" in q and ("bl1" in q or "b1" in q) and ("bl2" in q or "b2" in q):
        add(
            "SELECT ?marketSegment "
            "(SUM(IF(?baseline = \"BL1\", ?pct, 0)) AS ?changeBL1) "
            "(SUM(IF(?baseline = \"BL2\", ?pct, 0)) AS ?changeBL2) WHERE { "
            "survey:Tier1CurrentDemand a survey:CurrentDemandAnalysis ; "
            "survey:hasSurveyOrigin survey:Tier1_Survey ; "
            "survey:hasMarketSegment survey:Automotive ; "
            "survey:hasAggregatedResult ?entry . "
            "?entry a survey:CurrentDemandAnalysis ; "
            "survey:hasMarketSegment survey:Automotive ; "
            "survey:baselineType ?baseline ; "
            "survey:percentageChange ?pct . "
            "FILTER(?baseline IN (\"BL1\", \"BL2\")) "
            "BIND(\"Automotive\" AS ?marketSegment) "
            "} GROUP BY ?marketSegment"
        )

    if (
        "automotive" in q
        and "future" in q
        and "demand" in q
        and ("option" in q or "options" in q)
        and "quarter" in q
    ):
        add(
            "SELECT ?quarter "
            "(SUM(IF(?baseline = \"Option1\", ?pct, 0)) AS ?Option1) "
            "(SUM(IF(?baseline = \"Option2\", ?pct, 0)) AS ?Option2) "
            "(SUM(IF(?baseline = \"Option3\", ?pct, 0)) AS ?Option3) WHERE { "
            "?entry a survey:FutureDemandAnalysis ; "
            "survey:hasMarketSegment survey:Automotive ; "
            "survey:baselineType ?baseline ; "
            "survey:forTimePeriod ?quarter ; "
            "survey:percentageChange ?pct . "
            "FILTER(?baseline IN (\"Option1\", \"Option2\", \"Option3\")) "
            "} GROUP BY ?quarter ORDER BY ?quarter"
        )

    if (
        "oem" in q
        and "current" in q
        and "demand" in q
        and "percentage" in q
        and ("vehicle" in q or "bev" in q or "behv" in q or "ice" in q)
    ):
        add(
            "SELECT ?vehicle ?pct WHERE { "
            "?agg a survey:AggregatedDemand ; "
            "survey:hasSurveyOrigin survey:OEM_Survey_Instance ; "
            "survey:isAggregatedResult true ; "
            "survey:forVehicleType ?veh ; "
            "survey:percentageChange ?pct . "
            "BIND(IF(?veh = survey:BEV, \"BEV\", IF(?veh = survey:BEHV, \"BEHV\", IF(?veh = survey:ICE, \"ICE\", \"\"))) AS ?vehicle) "
            "FILTER(?vehicle != \"\") "
            "BIND(IF(?vehicle = \"BEV\", 1, IF(?vehicle = \"BEHV\", 2, IF(?vehicle = \"ICE\", 3, 4))) AS ?ord) "
            "} ORDER BY ?ord"
        )

    if (
        "oem" in q
        and "autonomous" in q
        and "vehicle" in q
        and "sae" in q
        and "year" in q
    ):
        add(
            "SELECT ?vehicle ?saeLabel ?year (AVG(?pct) AS ?avgPct) WHERE { "
            "?oemClass a survey:AutonomousDrivingDevelopment_OEM ; "
            "survey:hasSurveyOrigin survey:OEM_Survey ; "
            "survey:hasDetail ?entry . "
            "?entry a survey:AutonomousDrivingDevelopment ; "
            "survey:hasVehicleType ?veh ; "
            "survey:hasSAELevel ?sae ; "
            "survey:hasPercentage ?pct ; "
            "survey:hasYear ?year . "
            "BIND(IF(CONTAINS(STR(?veh), \"BEV\"), \"BEV\", IF(CONTAINS(STR(?veh), \"BEHV\"), \"BEHV\", IF(CONTAINS(STR(?veh), \"ICE\"), \"ICE\", \"OTHER\"))) AS ?vehicle) "
            "FILTER(?vehicle != \"OTHER\") "
            "BIND(STRAFTER(STR(?sae), \"SAE_Level_\") AS ?saeLabel) "
            "} GROUP BY ?vehicle ?saeLabel ?year ORDER BY ?vehicle xsd:integer(?saeLabel) xsd:integer(?year)"
        )

    if (
        ("tier1" in q or "tier 1" in q or "oem" in q)
        and "autonomous" in q
        and "vehicle" in q
        and "sae" in q
        and "year" in q
    ):
        origin = "survey:Tier1_Survey" if ("tier1" in q or "tier 1" in q) else "survey:OEM_Survey"
        detail_class = (
            "survey:AutonomousDrivingDevelopment_Tier1"
            if ("tier1" in q or "tier 1" in q)
            else "survey:AutonomousDrivingDevelopment_OEM"
        )
        add(
            "SELECT ?vehicle ?saeLabel ?year (AVG(?pct) AS ?avgPct) WHERE { "
            f"?holder a {detail_class} ; "
            f"survey:hasSurveyOrigin {origin} ; "
            "survey:hasDetail ?entry . "
            "?entry a survey:AutonomousDrivingDevelopment ; "
            "survey:hasVehicleType ?veh ; "
            "survey:hasSAELevel ?sae ; "
            "survey:hasPercentage ?pct ; "
            "survey:hasYear ?year . "
            "BIND(REPLACE(STR(?veh), '^.*/', '') AS ?vehicle) "
            "BIND(REPLACE(STR(?sae), '^.*/SAE_Level_', '') AS ?saeLabel) "
            "} GROUP BY ?vehicle ?saeLabel ?year ORDER BY ?vehicle xsd:integer(?saeLabel) xsd:integer(?year)"
        )

    if (
        "future" in q
        and "demand" in q
        and "share" in q
        and "quarter" in q
        and "region" in q
        and ("oem" in q or "semiconductor" in q or "semi" in q)
    ):
        origin = "survey:OEM_Survey" if "oem" in q else "survey:Semiconductor_Survey"
        add(
            "SELECT ?quarterLabel ?regionName "
            "(ROUND(((SUM(?pct) / ?grandTotal) * 100) * 100) / 100 AS ?percentShare) WHERE { "
            f"BIND({origin} AS ?surveyType) "
            "?origin a ?surveyType . "
            "?entry a survey:DemandForRegion ; "
            "survey:hasSurveyOrigin ?origin ; "
            "survey:inRegion ?region ; "
            "survey:quarter ?q ; "
            "survey:totalDemandPercentageChange ?pct . "
            "?q survey:periodLabel ?quarterLabel . "
            "?region survey:regionName ?regionName . "
            "{ SELECT ?origin ?quarterLabel (SUM(?v) AS ?grandTotal) WHERE { "
            "?e a survey:DemandForRegion ; "
            "survey:hasSurveyOrigin ?origin ; "
            "survey:quarter ?qq ; "
            "survey:totalDemandPercentageChange ?v . "
            "?qq survey:periodLabel ?quarterLabel . "
            "} GROUP BY ?origin ?quarterLabel } "
            "} GROUP BY ?quarterLabel ?regionName ?grandTotal ORDER BY ?quarterLabel ?regionName"
        )

    if "shortage" in q and "group" in q and "survey origin" in q:
        add(
            "SELECT ?surveyType ?shortageStatus (COUNT(?company) AS ?companyCount) WHERE { "
            f"{survey_values}"
            "?company a survey:Company ; "
            "survey:hasSurveyOrigin ?origin ; "
            "survey:reportsShortage ?shortage . "
            "?origin a ?surveyClass . "
            "BIND(IF(?shortage = true, 'yes', 'no') AS ?shortageStatus) "
            "} GROUP BY ?surveyType ?shortageStatus ORDER BY ?surveyType ?shortageStatus"
        )

    if (
        "shortage" in q
        and "compan" in q
        and any(w in q for w in ["which", "list", "show", "indicated", "reported", "mentioned", "facing"])
        and not any(w in q for w in ["how many", "count", "number", "versus", "vs"])
    ):
        add(
            "SELECT ?companyName ?surveyType WHERE { "
            f"{survey_values}"
            "?company a survey:Company ; "
            "survey:companyName ?companyName ; "
            "survey:hasSurveyOrigin ?origin ; "
            "survey:reportsShortage true . "
            "?origin a ?surveyClass . "
            "} ORDER BY ?surveyType ?companyName"
        )

    if "shortage" in q and "split" in q and any(w in q for w in ["semiconductor", "semi", "oem", "tier1", "tier 1"]):
        if "oem" in q:
            origin = "survey:OEM_Survey"
        elif "tier1" in q or "tier 1" in q:
            origin = "survey:Tier1_Survey"
        else:
            origin = "survey:Semiconductor_Survey"
        add(
            "SELECT ?status (COUNT(?c) AS ?count) WHERE { "
            "?c a survey:Company ; survey:hasSurveyOrigin ?o ; survey:reportsShortage ?flag . "
            f"?o a {origin} . "
            "BIND(IF(?flag = true, 'yes', 'no') AS ?status) "
            "} GROUP BY ?status ORDER BY ?status"
        )

    if (
        "shortage" in q
        and any(w in q for w in ["versus", "no shortage", "yes or no", "each shortage response", "by shortage"])
        and any(w in q for w in ["semiconductor", "semi", "oem", "tier1", "tier 1"])
    ):
        if "oem" in q:
            origin = "survey:OEM_Survey"
        elif "tier1" in q or "tier 1" in q:
            origin = "survey:Tier1_Survey"
        else:
            origin = "survey:Semiconductor_Survey"
        add(
            "SELECT ?ShortageStatus (COUNT(?Company) AS ?Count) WHERE { "
            "?Company a survey:Company ; "
            "survey:hasSurveyOrigin ?origin ; "
            "survey:reportsShortage ?Shortage . "
            f"?origin a {origin} . "
            "BIND(IF(?Shortage = true, \"yes\", \"no\") AS ?ShortageStatus) "
            "} GROUP BY ?ShortageStatus"
        )

    if "shortage" in q and any(w in q for w in ["compare", "status"]) and "group" not in q:
        add(
            "SELECT ?surveyType ?status (COUNT(?c) AS ?count) WHERE { "
            "?c a survey:Company ; survey:hasSurveyOrigin ?o ; survey:reportsShortage ?flag . "
            "?o a ?surveyType . "
            "FILTER(?surveyType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey)) "
            "BIND(IF(?flag = true, 'yes', 'no') AS ?status) "
            "} GROUP BY ?surveyType ?status ORDER BY ?surveyType ?status"
        )

    if (
        "shortage" in q
        and any(w in q for w in ["how many", "count"])
        and "group" not in q
        and not any(w in q for w in ["versus", "no shortage", "yes or no", "each shortage response", "by shortage"])
    ):
        origin = None
        if "oem" in q:
            origin = "survey:OEM_Survey"
        elif "tier1" in q or "tier 1" in q:
            origin = "survey:Tier1_Survey"
        elif "semiconductor" in q or "semi" in q:
            origin = "survey:Semiconductor_Survey"
        if origin:
            add(
                "SELECT (COUNT(?c) AS ?count) WHERE { "
                "?c a survey:Company ; survey:hasSurveyOrigin ?o ; survey:reportsShortage ?flag . "
                f"?o a {origin} . "
                "FILTER(?flag = true) "
                "}"
            )

    if "demand" in q and "region" in q and any(
        w in q for w in ["tier1", "tier 1", "oem", "semiconductor", "semi"]
    ):
        if (
            "quarter" in q
            or "trend" in q
            or "percentage change" in q
            or ("future" in q and "percentage" in q)
        ):
            if "tier1" in q or "tier 1" in q:
                origin = "survey:Tier1_Survey"
            elif "oem" in q:
                origin = "survey:OEM_Survey"
            elif "semiconductor" in q or "semi" in q:
                origin = "survey:Semiconductor_Survey"
            else:
                origin = None
            if origin:
                add(
                    "SELECT ?quarterLabel ?regionName (SUM(?pct) AS ?totalPctChange) WHERE { "
                    "?d a survey:DemandForRegion ; "
                    "survey:hasSurveyOrigin ?o ; "
                    "survey:inRegion ?r ; "
                    "survey:quarter ?q ; "
                    "survey:totalDemandPercentageChange ?pct . "
                    f"?o a {origin} . "
                    "?r survey:regionName ?regionName . "
                    "?q survey:periodLabel ?quarterLabel . "
                    "} GROUP BY ?quarterLabel ?regionName ORDER BY ?quarterLabel ?regionName"
                )
                add(
                    "SELECT ?quarterLabel ?regionName (SUM(?pct) AS ?totalPctChange) WHERE { "
                    "?d a survey:DemandForRegion ; "
                    "survey:hasSurveyOrigin ?o ; "
                    "survey:inRegion ?r ; "
                    "survey:quarter ?q ; "
                    "survey:totalDemandPercentageChange ?pct . "
                    f"?o a {origin} . "
                    "?r survey:regionName ?regionName . "
                    "BIND(REPLACE(STR(?q), '^.*/', '') AS ?quarterLabel) "
                    "} GROUP BY ?quarterLabel ?regionName ORDER BY ?quarterLabel ?regionName"
                )
        elif "oem" in q and ("tier1" in q or "tier 1" in q) and "semiconductor" in q:
            total_demand_query = (
                "SELECT ?surveyType ?regionName (SUM(?units) AS ?totalDemand) WHERE { "
                "?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:inRegion ?r ; survey:totalDemand ?units . "
                "?o a ?surveyType . "
                "?r survey:regionName ?regionName . "
                "FILTER(?surveyType IN (survey:Tier1_Survey, survey:OEM_Survey, survey:Semiconductor_Survey)) "
                "} GROUP BY ?surveyType ?regionName ORDER BY ?surveyType ?regionName"
            )
            pct_change_query = (
                "SELECT ?regionName ?surveyType (SUM(?unitsSold) AS ?totalDemand) WHERE { "
                f"{survey_values}"
                "?demandForRegion a survey:DemandForRegion ; "
                "survey:hasSurveyOrigin ?origin ; "
                "survey:inRegion ?region ; "
                "survey:totalDemandPercentageChange ?unitsSold . "
                "?origin a ?surveyClass . "
                "?region a survey:Region ; survey:regionName ?regionName . "
                "} GROUP BY ?regionName ?surveyType ORDER BY ?regionName ?surveyType"
            )
            if "total demand" in q or "across regions" in q:
                add(total_demand_query)
                add(pct_change_query)
            else:
                add(pct_change_query)
                add(total_demand_query)
        elif "oem" in q and ("tier1" in q or "tier 1" in q):
            add(
                "SELECT ?regionName "
                "(SUM(IF(?surveyClass = survey:Tier1_Survey, ?unitsSold, 0)) AS ?tier1Demand) "
                "(SUM(IF(?surveyClass = survey:OEM_Survey, ?unitsSold, 0)) AS ?oemDemand) WHERE { "
                "VALUES ?surveyClass { survey:Tier1_Survey survey:OEM_Survey } "
                "?demandForRegion a survey:DemandForRegion ; "
                "survey:hasSurveyOrigin ?origin ; "
                "survey:inRegion ?region ; "
                "survey:totalDemandPercentageChange ?unitsSold . "
                "?origin a ?surveyClass . "
                "?region a survey:Region ; survey:regionName ?regionName . "
                "} GROUP BY ?regionName ORDER BY ?regionName"
            )
        elif (
            ("semiconductor" in q or "semi" in q)
            and any(w in q for w in ["top", "highest", "largest"])
            and not any(w in q for w in ["total demand", "current demand", "demand totals"])
        ):
            add(
                "SELECT ?regionName (SUM(?unitsSold) AS ?semiconductorDemand) WHERE { "
                "?demandForRegion a survey:DemandForRegion ; "
                "survey:hasSurveyOrigin ?origin ; "
                "survey:inRegion ?region ; "
                "survey:totalDemandPercentageChange ?unitsSold . "
                "?origin a survey:Semiconductor_Survey . "
                "?region a survey:Region ; survey:regionName ?regionName . "
                "} GROUP BY ?regionName ORDER BY DESC(?semiconductorDemand) LIMIT 1"
            )
        elif any(w in q for w in ["total demand", "current demand", "aggregate total demand", "demand totals"]):
            if "oem" in q:
                origin = "survey:OEM_Survey"
            elif "tier1" in q or "tier 1" in q:
                origin = "survey:Tier1_Survey"
            elif "semiconductor" in q or "semi" in q:
                origin = "survey:Semiconductor_Survey"
            else:
                origin = None
            if origin:
                add(
                    "SELECT ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE { "
                    "?demandForRegion a survey:DemandForRegion ; "
                    "survey:hasSurveyOrigin ?origin ; "
                    "survey:inRegion ?region ; "
                    "survey:totalDemand ?unitsSold . "
                    f"?origin a {origin} . "
                    "?region a survey:Region ; survey:regionName ?regionName . "
                    "} GROUP BY ?regionName ORDER BY DESC(?totalDemand)"
                )

    if (
        "demand" in q
        and "region" in q
        and not any(w in q for w in ["tier1", "tier 1", "oem", "semiconductor", "semi"])
        and not any(w in q for w in ["quarter", "future", "percentage change"])
    ):
        add(
            "SELECT ?surveyType ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE { "
            f"{survey_values}"
            "?demandForRegion a survey:DemandForRegion ; "
            "survey:hasSurveyOrigin ?origin ; "
            "survey:inRegion ?region ; "
            "survey:totalDemand ?unitsSold . "
            "?origin a ?surveyClass . "
            "?region a survey:Region ; survey:regionName ?regionName . "
            "} GROUP BY ?surveyType ?regionName ORDER BY ?surveyType ?regionName"
        )
        add(
            "SELECT ?regionName (SUM(?unitsSold) AS ?totalDemand) WHERE { "
            "?demandForRegion a survey:DemandForRegion ; "
            "survey:inRegion ?region ; "
            "survey:totalDemand ?unitsSold . "
            "?region a survey:Region ; survey:regionName ?regionName . "
            "} GROUP BY ?regionName ORDER BY DESC(?totalDemand)"
        )

    if (
        "demand" in q
        and "region" in q
        and any(w in q for w in ["survey-origin", "survey origin", "origin class", "survey group", "survey groups"])
    ):
        add(
            "SELECT ?regionName ?surveyType (SUM(?unitsSold) AS ?totalDemand) WHERE { "
            f"{survey_values}"
            "?demandForRegion a survey:DemandForRegion ; "
            "survey:hasSurveyOrigin ?origin ; "
            "survey:inRegion ?region ; "
            "survey:totalDemand ?unitsSold . "
            "?origin a ?surveyClass . "
            "?region a survey:Region ; survey:regionName ?regionName . "
            "} GROUP BY ?regionName ?surveyType ORDER BY ?regionName ?surveyType"
        )

    if "origin type" in q and "demand" in q:
        add(
            "SELECT ?surveyType (SUM(?unitsSold) AS ?totalDemand) WHERE { "
            f"{survey_values}"
            "?demandForRegion a survey:DemandForRegion ; "
            "survey:hasSurveyOrigin ?origin ; "
            "survey:totalDemandPercentageChange ?unitsSold . "
            "?origin a ?surveyClass . "
            "} GROUP BY ?surveyType ORDER BY DESC(?totalDemand) LIMIT 1"
        )

    if "average" in q and "oem" in q and "demand" in q and "japan" in q:
        add(
            "SELECT ?regionName (AVG(?units) AS ?avgDemand) WHERE { "
            "?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:inRegion ?r ; survey:totalDemand ?units . "
            "?o a survey:OEM_Survey . "
            "?r survey:regionName ?regionName . "
            "FILTER(?regionName = 'Japan') "
            "} GROUP BY ?regionName"
        )

    if "future" in q and "demand" in q and "technology" in q and "quarter" in q:
        add(
            "SELECT ?techLabel ?quarterLabel (SUM(?pct) AS ?totalFutureChange) WHERE { "
            "?entry a survey:FutureDemandAnalysis ; "
            "survey:analyzesTechnologyCategory ?tech ; "
            "survey:forTimePeriod ?quarter ; "
            "survey:percentageChange ?pct . "
            "BIND(REPLACE(STR(?tech), '^.*/', '') AS ?techLabel) "
            "BIND(REPLACE(STR(?quarter), '^.*/', '') AS ?quarterLabel) "
            "} GROUP BY ?techLabel ?quarterLabel ORDER BY ?techLabel ?quarterLabel"
        )
        if any(w in q for w in ["semiconductor", "semi", "tier1", "tier 1", "oem"]):
            if "oem" in q:
                origin = "survey:OEM_Survey"
            elif "tier1" in q or "tier 1" in q:
                origin = "survey:Tier1_Survey"
            else:
                origin = "survey:Semiconductor_Survey"
            add(
                "SELECT ?techLabel ?quarterLabel (SUM(?pct) AS ?totalFutureChange) WHERE { "
                "?entry a survey:FutureDemandAnalysis ; "
                "survey:hasSurveyOrigin ?origin ; "
                "survey:analyzesTechnologyCategory ?tech ; "
                "survey:forTimePeriod ?quarter ; "
                "survey:percentageChange ?pct . "
                f"?origin a {origin} . "
                "BIND(REPLACE(STR(?tech), '^.*/', '') AS ?techLabel) "
                "BIND(REPLACE(STR(?quarter), '^.*/', '') AS ?quarterLabel) "
                "} GROUP BY ?techLabel ?quarterLabel ORDER BY ?techLabel ?quarterLabel"
            )

    if "future" in q and "demand" in q and "vehicle" in q and "quarter" in q:
        add(
            "SELECT ?vehicleType ?quarterLabel (AVG(?pct) AS ?avgChange) WHERE { "
            "?entry a survey:FutureDemandAnalysis ; "
            "survey:analyzesVehicleType ?vehicle ; "
            "survey:forTimePeriod ?quarter ; "
            "survey:percentageChange ?pct . "
            "BIND(REPLACE(STR(?vehicle), '^.*/', '') AS ?vehicleType) "
            "BIND(REPLACE(STR(?quarter), '^.*/', '') AS ?quarterLabel) "
            "} GROUP BY ?vehicleType ?quarterLabel ORDER BY ?vehicleType ?quarterLabel"
        )

    if (
        "autonomous" in q
        and "vehicle" in q
        and any(w in q for w in ["year", "yearly", "across years", "over the years"])
        and any(w in q for w in ["average", "avg", "mean", "percentages", "percentage"])
    ):
        add(
            "SELECT ?year ?vehicleType (AVG(?percentage) AS ?avgPercentage) WHERE { "
            "?entry a survey:AutonomousDrivingDevelopment ; "
            "survey:hasVehicleType ?vehicle ; "
            "survey:hasSAELevel ?level ; "
            "survey:hasYear ?year ; "
            "survey:hasPercentage ?percentage . "
            "BIND(REPLACE(STR(?vehicle), '^.*/', '') AS ?vehicleType) "
            "} GROUP BY ?year ?vehicleType ORDER BY ?year ?vehicleType"
        )

    if (
        ("autonomous" in q or "sae" in q)
        and any(w in q for w in ["vehicle", "category", "type"])
        and any(w in q for w in ["level 5", "sae level 5", "full autonomy", "fully autonomous"])
        and any(w in q for w in ["highest", "strongest", "largest", "top"])
    ):
        add(
            "SELECT ?vehicleType ?percentage WHERE { "
            "?entry a survey:AutonomousDrivingDevelopment ; "
            "survey:hasVehicleType ?vehicle ; "
            "survey:hasSAELevel survey:SAE_Level_5 ; "
            "survey:hasPercentage ?percentage . "
            "BIND(REPLACE(STR(?vehicle), '^.*/', '') AS ?vehicleType) "
            "} ORDER BY DESC(?percentage) LIMIT 1"
        )

    if (
        "autonomous" in q
        and "vehicle" in q
        and "sae" in q
        and any(w in q for w in ["highest", "max", "maximum", "top"])
    ):
        add(
            "SELECT ?vehicleType ?saeLevel (MAX(?percentage) AS ?maxPercentage) WHERE { "
            "?entry a survey:AutonomousDrivingDevelopment ; "
            "survey:hasVehicleType ?vehicle ; "
            "survey:hasSAELevel ?sae ; "
            "survey:hasPercentage ?percentage . "
            "BIND(REPLACE(STR(?vehicle), '^.*/', '') AS ?vehicleType) "
            "BIND(REPLACE(STR(?sae), '^.*/SAE_Level_', '') AS ?saeLevel) "
            "} GROUP BY ?vehicleType ?saeLevel ORDER BY DESC(?maxPercentage) LIMIT 1"
        )

    if "autonomous" in q and "vehicle" in q and "sae" in q and "percentage" in q:
        add(
            "SELECT ?vehicleType ?saeLevel (AVG(?percentage) AS ?avgPercentage) WHERE { "
            "?entry a survey:AutonomousDrivingDevelopment ; "
            "survey:hasVehicleType ?vehicle ; "
            "survey:hasSAELevel ?sae ; "
            "survey:hasPercentage ?percentage . "
            "BIND(REPLACE(STR(?vehicle), '^.*/', '') AS ?vehicleType) "
            "BIND(REPLACE(STR(?sae), '^.*/SAE_Level_', '') AS ?saeLevel) "
            "} GROUP BY ?vehicleType ?saeLevel ORDER BY ?vehicleType xsd:integer(?saeLevel)"
        )

    if asks_order_cancellation and "response type" in q and any(
        w in q for w in ["highest", "largest", "top", "most", "overall"]
    ):
        add(
            "SELECT ?responseType (SUM(xsd:integer(?participants)) AS ?participantCount) WHERE { "
            "?entry a survey:OrderCancellation ; "
            "survey:hasResponseType ?responseType ; "
            "survey:participantCount ?participants . "
            "} GROUP BY ?responseType ORDER BY DESC(?participantCount) LIMIT 1"
        )

    if asks_order_cancellation and "technology" in q and asks_count_or_sum and "response type" not in q:
        add(
            "SELECT ?technologyCategory (SUM(xsd:integer(?participants)) AS ?participantCount) WHERE { "
            "?entry a survey:OrderCancellation ; "
            "survey:forTechnologyCategory ?tech ; "
            "survey:participantCount ?participants . "
            "BIND(REPLACE(STR(?tech), '^.*/', '') AS ?technologyCategory) "
            "} GROUP BY ?technologyCategory ORDER BY ?technologyCategory"
        )

    if asks_order_cancellation and "technology" in q:
        add(
            "SELECT ?technologyCategory ?responseType (SUM(xsd:integer(?participants)) AS ?participantCount) WHERE { "
            "?entry a survey:OrderCancellation ; "
            "survey:forTechnologyCategory ?tech ; "
            "survey:hasResponseType ?responseType ; "
            "survey:participantCount ?participants . "
            "BIND(REPLACE(STR(?tech), '^.*/', '') AS ?technologyCategory) "
            "} GROUP BY ?technologyCategory ?responseType ORDER BY ?technologyCategory ?responseType"
        )
        add(
            "SELECT ?technologyCategory ?responseType (SUM(?participants) AS ?participantCount) WHERE { "
            "?entry a survey:OrderCancellation ; "
            "survey:forTechnologyCategory ?tech ; "
            "survey:hasResponseType ?responseType ; "
            "survey:participantCount ?participants . "
            "BIND(REPLACE(STR(?tech), '^.*/', '') AS ?technologyCategory) "
            "} GROUP BY ?technologyCategory ?responseType ORDER BY ?technologyCategory ?responseType"
        )

    if "inventory" in q and ("tier1" in q or "tier 1" in q):
        if any(w in q for w in ["participant", "participants", "total", "amount", "amounts", "overall"]):
            if "component" in q and "trend" not in q:
                add(
                    "SELECT ?componentLabel (SUM(xsd:integer(?participants)) AS ?participantCount) WHERE { "
                    "?entry a survey:InventoryDevelopment_Tier1 ; "
                    "survey:forComponent ?component ; "
                    "survey:participantCount ?participants . "
                    "BIND(REPLACE(STR(?component), '^.*/', '') AS ?componentLabel) "
                    "} GROUP BY ?componentLabel ORDER BY DESC(?participantCount)"
                )
            add(
                "SELECT ?componentLabel ?trend (SUM(xsd:integer(?participants)) AS ?participantCount) WHERE { "
                "?entry a survey:InventoryDevelopment_Tier1 ; "
                "survey:forComponent ?component ; "
                "survey:inventoryTrend ?trend ; "
                "survey:participantCount ?participants . "
                "BIND(REPLACE(STR(?component), '^.*/', '') AS ?componentLabel) "
                "} GROUP BY ?componentLabel ?trend ORDER BY ?componentLabel ?trend"
            )
            add(
                "SELECT ?componentLabel (SUM(xsd:integer(?participants)) AS ?participantCount) WHERE { "
                "?entry a survey:InventoryDevelopment_Tier1 ; "
                "survey:forComponent ?component ; "
                "survey:inventoryTrend ?trend ; "
                "survey:participantCount ?participants . "
                "BIND(REPLACE(STR(?component), '^.*/', '') AS ?componentLabel) "
                "} GROUP BY ?componentLabel ORDER BY DESC(?participantCount)"
            )
        add(
            "SELECT ?componentLabel ?trend (COUNT(?entry) AS ?entryCount) WHERE { "
            "?entry a survey:InventoryDevelopment_Tier1 ; "
            "survey:forComponent ?component ; "
            "survey:inventoryTrend ?trend . "
            "BIND(REPLACE(STR(?component), '^.*/', '') AS ?componentLabel) "
            "} GROUP BY ?componentLabel ?trend ORDER BY ?componentLabel ?trend"
        )

    if "inventory" in q and ("semiconductor" in q or "semi" in q):
        add(
            "SELECT ?technologyCategory ?trend (COUNT(?entry) AS ?entryCount) WHERE { "
            "?entry a survey:InventoryDevelopment_Semi ; "
            "survey:forTechnologyCategory ?tech ; "
            "survey:hasInventoryTrend ?trend . "
            "BIND(REPLACE(STR(?tech), '^.*/', '') AS ?technologyCategory) "
            "} GROUP BY ?technologyCategory ?trend ORDER BY ?technologyCategory ?trend"
        )
        if "most" in q or "highest" in q or "common" in q or "pairing" in q:
            add(
                "SELECT ?technologyCategory ?trend (COUNT(?entry) AS ?entryCount) WHERE { "
                "?entry a survey:InventoryDevelopment_Semi ; "
                "survey:forTechnologyCategory ?tech ; "
                "survey:hasInventoryTrend ?trend . "
                "BIND(REPLACE(STR(?tech), '^.*/', '') AS ?technologyCategory) "
                "} GROUP BY ?technologyCategory ?trend ORDER BY DESC(?entryCount) LIMIT 1"
            )

    if "future" in q and "demand" in q and "tech" in q and any(w in q for w in ["strongest", "largest", "highest"]):
        add(
            "SELECT ?techLabel (AVG(?pct) AS ?avgFutureChange) WHERE { "
            "?entry a survey:FutureDemandAnalysis ; "
            "survey:analyzesTechnologyCategory ?tech ; "
            "survey:percentageChange ?pct . "
            "BIND(REPLACE(STR(?tech), '^.*/', '') AS ?techLabel) "
            "} GROUP BY ?techLabel ORDER BY DESC(?avgFutureChange) LIMIT 1"
        )

    if "tier1" in q and ("bl1" in q or "b1" in q) and ("bl2" in q or "b2" in q):
        if "minus" in q or "delta" in q:
            add(
                "SELECT ((SUM(IF(?baseline = 'BL1', ?pct, 0)) - SUM(IF(?baseline = 'BL2', ?pct, 0))) AS ?deltaBL1BL2) WHERE { "
                "survey:Tier1CurrentDemand a survey:CurrentDemandAnalysis ; survey:hasAggregatedResult ?entry . "
                "?entry a survey:CurrentDemandAnalysis ; survey:baselineType ?baseline ; survey:percentageChange ?pct . "
                "FILTER(?baseline IN ('BL1','BL2')) "
                "}"
            )
        else:
            add(
                "SELECT ?baseline (SUM(?pct) AS ?totalChange) WHERE { "
                "survey:Tier1CurrentDemand a survey:CurrentDemandAnalysis ; survey:hasAggregatedResult ?entry . "
                "?entry a survey:CurrentDemandAnalysis ; survey:baselineType ?baseline ; survey:percentageChange ?pct . "
                "FILTER(?baseline IN ('BL1','BL2')) "
                "} GROUP BY ?baseline ORDER BY ?baseline"
            )

    if "tech" in q and "current demand" in q and ("b1" in q or "bl1" in q) and ("b2" in q or "bl2" in q):
        add(
            "SELECT ?technologyCategory ?currentDemand ?percentChangeB1 ?percentChangeB2 WHERE { "
            "?tech a survey:TechnologyCategory ; "
            "survey:technologyCategoryName ?technologyCategory ; "
            "survey:currentDemand ?currentDemand ; "
            "survey:percentChangeB1 ?percentChangeB1 ; "
            "survey:percentChangeB2 ?percentChangeB2 . "
            "} ORDER BY ?technologyCategory"
        )

    if "technology" in q and "current demand" in q and any(w in q for w in ["largest", "highest", "top"]):
        add(
            "SELECT ?technologyCategory ?currentDemand WHERE { "
            "?tech a survey:TechnologyCategory ; "
            "survey:technologyCategoryName ?technologyCategory ; "
            "survey:currentDemand ?currentDemand . "
            "} ORDER BY DESC(?currentDemand) LIMIT 1"
        )

    if "conversion" in q and "technology" in q and "vehicle" in q:
        add(
            "SELECT ?technologyCategory ?vehicleType ?conversionValue WHERE { "
            "?tech a survey:TechnologyCategory ; "
            "survey:technologyCategoryName ?technologyCategory ; "
            "survey:hasConversionFactor ?factor . "
            "?factor a survey:ConversionFactor ; "
            "survey:appliesToVehicleType ?vehicle ; "
            "survey:conversionValue ?conversionValue . "
            "BIND(REPLACE(STR(?vehicle), '^.*/', '') AS ?vehicleType) "
            "} ORDER BY ?technologyCategory ?vehicleType"
        )

    if asks_forecast_sales and asks_monthly_sales:
        add(
            "SELECT ?month (SUM(?unitsSold) AS ?forecastUnits) WHERE { "
            "?obs a survey:VehicleSalesObservation ; "
            "survey:isForecastData true ; "
            "survey:forTimePeriod ?period ; "
            "survey:unitsSold ?unitsSold . "
            "BIND(REPLACE(STR(?period), '^.*/', '') AS ?month) "
            "} GROUP BY ?month ORDER BY ?month"
        )

    if asks_actual_sales and asks_forecast_sales and asks_monthly_sales:
        add(
            "SELECT ?monthLabel ?dataType (SUM(?unitsSold) AS ?totalUnitsSold) WHERE { "
            "?obs a survey:VehicleSalesObservation ; "
            "survey:forTimePeriod ?month ; "
            "survey:unitsSold ?unitsSold . "
            "OPTIONAL { ?obs survey:isActualData ?actual . } "
            "OPTIONAL { ?obs survey:isForecastData ?forecast . } "
            "BIND(IF(BOUND(?actual) && ?actual = true, 'actual', IF(BOUND(?forecast) && ?forecast = true, 'forecast', 'unknown')) AS ?dataType) "
            "BIND(REPLACE(STR(?month), '^.*/', '') AS ?monthLabel) "
            "} GROUP BY ?monthLabel ?dataType ORDER BY ?monthLabel ?dataType"
        )

    if asks_actual_sales and asks_forecast_sales:
        add(
            "SELECT ?dataType (COUNT(?obs) AS ?observationCount) WHERE { "
            "?obs a survey:VehicleSalesObservation . "
            "OPTIONAL { ?obs survey:isActualData ?actual . } "
            "OPTIONAL { ?obs survey:isForecastData ?forecast . } "
            "BIND(IF(BOUND(?actual) && ?actual = true, 'actual', IF(BOUND(?forecast) && ?forecast = true, 'forecast', 'unknown')) AS ?dataType) "
            "} GROUP BY ?dataType ORDER BY ?dataType"
        )

    if asks_actual_sales and asks_monthly_sales:
        add(
            "SELECT ?monthLabel (SUM(?units) AS ?unitsSold) WHERE { "
            "?obs a survey:VehicleSalesObservation ; "
            "survey:forTimePeriod ?month ; "
            "survey:isActualData true ; "
            "survey:unitsSold ?units . "
            "BIND(REPLACE(STR(?month), '^.*/', '') AS ?monthLabel) "
            "} GROUP BY ?monthLabel ORDER BY ?monthLabel"
        )

    if asks_yearly_sales or "yearly sales" in q or "annual sales" in q:
        if any(w in q for w in ["leads", "lead", "highest", "top", "largest", "most"]):
            add(
                "SELECT ?vehicleType (SUM(?sales) AS ?totalSales) WHERE { "
                "?entry a survey:YearlySalesData ; "
                "survey:analyzesVehicleType ?vehicle ; "
                "survey:yearlySales ?sales . "
                "BIND(REPLACE(STR(?vehicle), '^.*/', '') AS ?vehicleType) "
                "} GROUP BY ?vehicleType ORDER BY DESC(?totalSales) LIMIT 1"
            )
        else:
            add(
                "SELECT ?vehicleType ?year (SUM(?sales) AS ?yearlySales) WHERE { "
                "?entry a survey:YearlySalesData ; "
                "survey:analyzesVehicleType ?vehicle ; "
                "survey:forYear ?year ; "
                "survey:yearlySales ?sales . "
                "BIND(REPLACE(STR(?vehicle), '^.*/', '') AS ?vehicleType) "
                "} GROUP BY ?vehicleType ?year ORDER BY ?vehicleType ?year"
            )

    if "component-share" in q or "component share" in q:
        add(
            "SELECT ?companyName (COUNT(?share) AS ?activeCategories) WHERE { "
            "?share a survey:ComponentShare ; "
            "survey:forCompany ?company ; "
            "survey:isActiveInCategory true . "
            "?company survey:companyName ?companyName . "
            "} GROUP BY ?companyName ORDER BY DESC(?activeCategories)"
        )

    return templates


def repair_candidate_query(
    question: str,
    schema: KGSchema,
    invalid_query: str,
    error_message: str,
    llm_client: Optional[object] = None,
) -> Optional[str]:
    prompt = build_repair_prompt(
        question=question,
        schema=schema,
        invalid_query=invalid_query,
        error_message=error_message,
    )
    client = llm_client or _default_client()
    generated = client.generate(prompt, k=1)
    items = _normalize_generated_output(generated)
    if not items:
        return None
    repaired = _normalize_candidate_query(items[0])
    return repaired or None


def _candidate_validation_errors(query: str, schema: KGSchema) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    errors.extend(validate_query_syntax(query))
    errors.extend(validate_query_semantic(query))
    errors.extend(validate_query_schema(query, schema))
    if not errors:
        try:
            parseQuery(_ensure_prefixes(query))
        except Exception as exc:
            errors.append({"type": "syntax", "message": str(exc)})
    return errors


def _candidate_repair_feedback(question: str, query: str, schema: KGSchema) -> str:
    errors = _candidate_validation_errors(query, schema)
    messages = [
        f"{err.get('type', 'error')}: {err.get('message', '')}".strip()
        for err in errors
        if err.get("message")
    ]
    repair_mode = os.getenv("INFINEON_GENERATION_REPAIR_MODE", "invalid").strip().lower()
    if repair_mode in {"semantic", "intent", "coverage"}:
        coverage = semantic_coverage_report(question, query)
        missing = list(coverage.get("missing", []))
        if missing:
            messages.append(
                "intent: query does not cover requested concepts: "
                + ", ".join(missing[:8])
            )
    return "\n".join(messages)


def _repair_generated_candidates(
    question: str,
    schema: KGSchema,
    candidates: List[Dict[str, str]],
    seen: set,
    client: object,
    max_repairs: int,
) -> List[Dict[str, str]]:
    repaired_rows: List[Dict[str, str]] = []
    repair_mode = os.getenv("INFINEON_GENERATION_REPAIR_MODE", "invalid").strip().lower()
    allow_semantic_repair = repair_mode in {"semantic", "intent", "coverage"}

    for cand in candidates:
        if len(repaired_rows) >= max_repairs:
            break
        source = str(cand.get("source", "") or "")
        if source == "validated_retrieval":
            continue
        query = _normalize_candidate_query(str(cand.get("query", "")))
        if not query:
            continue
        errors = _candidate_validation_errors(query, schema)
        needs_repair = bool(errors)
        if allow_semantic_repair and not needs_repair:
            coverage = semantic_coverage_report(question, query)
            needs_repair = int(coverage.get("missing_count", 0)) > 0
        if not needs_repair:
            continue

        feedback = _candidate_repair_feedback(question, query, schema)
        if not feedback:
            continue
        try:
            repaired = repair_candidate_query(
                question=question,
                schema=schema,
                invalid_query=query,
                error_message=feedback,
                llm_client=client,
            )
        except Exception:
            continue
        repaired = _normalize_candidate_query(repaired or "")
        if not repaired:
            continue
        key = _candidate_key(repaired)
        if key in seen:
            continue
        repaired_errors = _candidate_validation_errors(repaired, schema)
        if repaired_errors:
            continue
        seen.add(key)
        repaired_rows.append(
            {
                "query": repaired,
                "source": "repair",
                "repair_source": source or "unknown",
                "repair_feedback": feedback,
            }
        )

    return repaired_rows


def generate_candidates(
    question: str,
    schema: KGSchema,
    k: int = 5,
    llm_client: Optional[object] = None,
    entity_alias_index: Optional[EntityAliasIndex] = None,
    entity_profile_graph: Optional[object] = None,
    max_entity_links: int = 5,
    query_plan_predictor: Optional[object] = None,
) -> Dict:
    resolved = canonicalize_question_with_index(
        question,
        index=entity_alias_index,
        max_matches=max_entity_links,
    )
    effective_question = resolved.effective_question
    entity_mappings = resolved.mappings
    linked_profiles: List[EntityProfile] = []
    if entity_alias_index and entity_profile_graph is not None and entity_mappings:
        matched_iris: List[str] = []
        for mapping in entity_mappings:
            key = normalize_alias(str(mapping.get("canonical", "")))
            matched_iris.extend(entity_alias_index.key_to_subjects.get(key, []))
        linked_profiles = build_profiles_for_iris(entity_profile_graph, matched_iris)
    predicted_query_plan_labels: List[str] = []
    if query_plan_predictor is not None:
        try:
            predicted_query_plan_labels = list(
                query_plan_predictor.predict_labels(effective_question)
            )
        except Exception:
            predicted_query_plan_labels = []

    slice_names: List[str] = []
    slice_route: Dict[str, object] = {"selected": [], "confidence": "disabled", "scores": []}
    sliced_schema: Optional[KGSchema] = None
    if schema_slicing_enabled():
        slice_route = infer_schema_slice_route(effective_question, predicted_query_plan_labels)
        slice_names = list(slice_route.get("selected") or [])
        if slice_names:
            sliced_schema = build_schema_slice(schema, slice_names)

    prompt_schema = sliced_schema or schema

    # Build prompt
    prompt = build_candidate_prompt(
        question=question,
        schema=prompt_schema,
        k=k,
        canonical_question=effective_question,
        entity_mappings=entity_mappings,
        entity_profiles=linked_profiles,
        predicted_query_plan_labels=predicted_query_plan_labels,
    )

    # Use provided client or default
    client = llm_client or _default_client()



    llm_cache_metadata: Dict[str, object] = {
        "llm_cache_enabled": _llm_cache_enabled(),
        "llm_cache_hit": False,
    }

    try:
        # Call LLM
        generated, llm_cache_metadata = _generate_with_cache(
            client,
            prompt,
            k=k,
            purpose="candidate_generation",
        )

        generated_full_schema: List[str] = []
        full_schema_cache_metadata: Dict[str, object] = {}
        if sliced_schema is not None and k > 1 and full_schema_fallback_enabled():
            full_prompt = build_candidate_prompt(
                question=question,
                schema=schema,
                k=k,
                canonical_question=effective_question,
                entity_mappings=entity_mappings,
                entity_profiles=linked_profiles,
                predicted_query_plan_labels=predicted_query_plan_labels,
            )
            generated_full_schema, full_schema_cache_metadata = _generate_with_cache(
                client,
                full_prompt,
                k=max(1, k // 2),
                purpose="candidate_generation_full_schema",
            )


        # -----------------------------
        # Normalize LLM output
        # -----------------------------
        generated = _normalize_generated_output(generated)


        seen = set()
        candidates = []
        generated_count = len(generated) + len(generated_full_schema)
        reserved_for_llm = min(2, generated_count)
        deterministic_budget = max(0, k - reserved_for_llm)

        for cand in _retrieve_validated_candidate_queries(effective_question, limit=min(3, deterministic_budget)):
            query = _normalize_candidate_query(str(cand.get("query", "")))
            if not query:
                continue
            key = _candidate_key(query)
            if key in seen:
                continue
            seen.add(key)
            row = dict(cand)
            row["query"] = query
            candidates.append(row)

        for query in _template_candidate_queries(effective_question):
            if len(candidates) >= deterministic_budget:
                break
            key = _candidate_key(query)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"query": query, "source": "template"})

        for text in generated:
            query = _normalize_candidate_query(str(text))
            if not query:
                continue
            key = _candidate_key(query)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"query": query, "source": "infineon"})
            if len(candidates) >= k:
                break

        for text in generated_full_schema:
            if len(candidates) >= k:
                break
            query = _normalize_candidate_query(str(text))
            if not query:
                continue
            key = _candidate_key(query)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"query": query, "source": "infineon_full_schema"})

        repair_count = 0
        repair_enabled = (
            os.getenv("INFINEON_ENABLE_GENERATION_REPAIR", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if repair_enabled and candidates:
            max_repairs = int(os.getenv("INFINEON_GENERATION_REPAIR_MAX", "2") or 2)
            repaired_rows = _repair_generated_candidates(
                question=effective_question,
                schema=schema,
                candidates=candidates,
                seen=seen,
                client=client,
                max_repairs=max(0, max_repairs),
            )
            if repaired_rows:
                candidates.extend(repaired_rows)
                repair_count = len(repaired_rows)

                protected_sources = {"validated_retrieval", "template", "repair"}
                while len(candidates) > k:
                    remove_idx = None
                    for idx in range(len(candidates) - 1, -1, -1):
                        if str(candidates[idx].get("source", "")) not in protected_sources:
                            remove_idx = idx
                            break
                    if remove_idx is None:
                        remove_idx = len(candidates) - 1
                    candidates.pop(remove_idx)

        if not candidates:
            print("⚠️ WARNING: No candidates generated!")

        return {
            "prompt": prompt,
            "candidates": candidates,
            "metadata": {
                "k_requested": k,
                "k_returned": len(candidates),
                "original_question": (question or "").strip(),
                "effective_question": effective_question,
                "entity_mappings": entity_mappings,
                "entity_profiles": [
                    {
                        "iri": profile.iri,
                        "canonical_label": profile.canonical_label,
                        "types": profile.types,
                        "predicates": profile.predicates,
                        "quality_flags": profile.quality_flags,
                    }
                    for profile in linked_profiles
                ],
                "entity_linking_applied": bool(entity_mappings),
                "predicted_query_plan_labels": predicted_query_plan_labels,
                "query_plan_predictor_applied": bool(predicted_query_plan_labels),
                "validated_retrieval_count": sum(
                    1 for cand in candidates if cand.get("source") == "validated_retrieval"
                ),
                "generation_repair_enabled": repair_enabled,
                "generation_repair_count": repair_count,
                "llm_cache_enabled": bool(llm_cache_metadata.get("llm_cache_enabled")),
                "llm_cache_hit": bool(llm_cache_metadata.get("llm_cache_hit")),
                "llm_cache_key": llm_cache_metadata.get("llm_cache_key"),
                "full_schema_llm_cache_hit": bool(full_schema_cache_metadata.get("llm_cache_hit", False)),
                "schema_slicing_applied": bool(sliced_schema),
                "schema_slice_names": slice_names,
                "schema_slice_confidence": str(slice_route.get("confidence", "disabled")),
                "schema_slice_scores": list(slice_route.get("scores") or []),
                "schema_slice_full_fallback_enabled": bool(full_schema_fallback_enabled()),
            },
        }

    except Exception as exc:
        print("\n❌ LLM GENERATION FAILED:")
        print(exc)
        raise
