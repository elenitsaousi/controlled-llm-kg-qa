# evaluate_unseen.py
# Evaluates system accuracy on unseen questions [7][8]
from dotenv import load_dotenv
load_dotenv(".env")
import os
import re
import json
import numpy as np
import joblib
from rdflib import Graph
from llm.candidate_generation import generate_candidates
from kg.schema import load_schema
from ranking.feature_extraction import extract_features
from ranking.feature_config import FEATURE_NAMES
from visualization.ambiguity_metrics import ambiguity_entropy

os.environ.setdefault("LLM_PROVIDER", "infineon")

# Load resources
schema = load_schema("data/infineon/schema.json")
with open("data/infineon/schema.json") as f:
    schema_dict = json.load(f)

g = Graph()
g.parse("data/infineon/graph.ttl", format="turtle")
print(f"Graph loaded: {len(g)} triples")

ranker_data = joblib.load('ranking/models/infineon_ranker.joblib')
model = ranker_data['model']
scaler = ranker_data['scaler']

with open("ranking/infineon_ambiguity_thresholds.json") as f:
    thresholds = json.load(f)
H1 = thresholds["H1"]
H2 = thresholds["H2"]

PREFIX = (
    "PREFIX survey: <http://www.semanticweb.org/gibajajulena/"
    "ontologies/2025/9/OEM_Monthly_Survey/>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)

# Gold queries [7 - 0.4.6]
gold_queries = {
    "LOW1": "SELECT (COUNT(DISTINCT ?r) AS ?count) WHERE { ?r a survey:Region }",
    "LOW2": "SELECT ?name WHERE { ?r a survey:Region ; survey:regionName ?name }",
    "LOW3": "SELECT (COUNT(?d) AS ?count) WHERE { ?d a survey:DemandForRegion }",
    "LOW5": "SELECT (COUNT(?c) AS ?count) WHERE { ?c a survey:Company }",
    "LOW6": "SELECT DISTINCT ?type WHERE { ?s a ?type . FILTER(?type IN (survey:OEM_Survey, survey:Tier1_Survey, survey:Semiconductor_Survey)) }",
    "LOW7": "SELECT (COUNT(?d) AS ?count) WHERE { ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o . ?o a survey:Tier1_Survey }",
    "LOW9": "SELECT (COUNT(?c) AS ?count) WHERE { ?c a survey:Company ; survey:hasSurveyOrigin ?o . ?o a survey:Tier1_Survey }",
    "LOW10": "SELECT (COUNT(?d) AS ?count) WHERE { ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o . ?o a survey:OEM_Survey }",
    "LOW11": "SELECT DISTINCT ?baseline WHERE { ?e survey:baselineType ?baseline }",
    "LOW15": "SELECT (COUNT(DISTINCT ?sae) AS ?count) WHERE { ?e survey:hasSAELevel ?sae }",
    "LOW17": "SELECT (COUNT(?i) AS ?count) WHERE { ?i a survey:InventoryDevelopment_Tier1 }",
    "LOW19": "SELECT (COUNT(?o) AS ?count) WHERE { ?o a survey:OrderCancellation }",
    "MID1": 'SELECT ?pct WHERE { survey:Tier1CurrentDemand survey:hasAggregatedResult ?entry . ?entry survey:baselineType ?baseline ; survey:percentageChange ?pct . FILTER(?baseline = "BL1") }',
    "MID5": 'SELECT ?name (SUM(?units) AS ?total) WHERE { ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:inRegion ?r ; survey:totalDemand ?units . ?o a survey:Tier1_Survey . ?r survey:regionName ?name . FILTER(?name = "Europe") } GROUP BY ?name',
    "MID6": "SELECT (COUNT(DISTINCT ?name) AS ?count) WHERE { ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:inRegion ?r . ?o a survey:Tier1_Survey . ?r survey:regionName ?name }",
    "MID7": "SELECT (AVG(?units) AS ?avg) WHERE { ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:totalDemand ?units . ?o a survey:Tier1_Survey }",
    "MID9": 'SELECT ?pct WHERE { survey:Tier1CurrentDemand survey:hasAggregatedResult ?entry . ?entry survey:baselineType ?baseline ; survey:percentageChange ?pct . FILTER(?baseline = "BL2") }',
    "MID10": 'SELECT ?status (COUNT(?c) AS ?count) WHERE { ?c a survey:Company ; survey:hasSurveyOrigin ?o ; survey:reportsShortage ?s . ?o a survey:Tier1_Survey . BIND(IF(?s = true, "yes", "no") AS ?status) } GROUP BY ?status',
    "MID11": "SELECT (SUM(?units) AS ?total) WHERE { ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:totalDemand ?units . ?o a survey:Tier1_Survey }",
    "MID12": "SELECT ?name (SUM(?units) AS ?total) WHERE { ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:inRegion ?r ; survey:totalDemand ?units . ?o a survey:Tier1_Survey . ?r survey:regionName ?name } GROUP BY ?name ORDER BY ASC(?total) LIMIT 1",
    "MID17": 'SELECT ?baseline ?pct WHERE { survey:Tier1CurrentDemand survey:hasAggregatedResult ?entry . ?entry survey:baselineType ?baseline ; survey:percentageChange ?pct . FILTER(?baseline IN ("BL1", "BL2")) }',
    "MID18": "SELECT ?name (SUM(?units) AS ?total) WHERE { ?d a survey:DemandForRegion ; survey:hasSurveyOrigin ?o ; survey:inRegion ?r ; survey:totalDemand ?units . ?o a survey:Tier1_Survey . ?r survey:regionName ?name } GROUP BY ?name ORDER BY DESC(?total) LIMIT 1",
    "HIGH5": "SELECT ?vehicle ?saeLabel ?year (AVG(?pct) AS ?avg) WHERE { ?root a survey:AutonomousDrivingDevelopment_Tier1 ; survey:hasSurveyOrigin survey:Tier1_Survey ; survey:hasDetail ?entry . ?entry survey:hasVehicleType ?veh ; survey:hasSAELevel ?sae ; survey:hasPercentage ?pct ; survey:hasYear ?year . BIND(STRAFTER(STR(?veh), 'survey:') AS ?vehicle) BIND(STRAFTER(STR(?sae), 'SAE_Level_') AS ?saeLabel) } GROUP BY ?vehicle ?saeLabel ?year ORDER BY ?vehicle ?saeLabel ?year",
    "HIGH10": 'SELECT ?baseline ?pct WHERE { survey:Tier1CurrentDemand survey:hasAggregatedResult ?entry . ?entry survey:baselineType ?baseline ; survey:percentageChange ?pct . FILTER(?baseline IN ("BL1", "BL2")) }',
}

# Load test questions
with open('untrained_test.json') as f:
    tests = json.load(f)

# Filter only questions with gold queries
test_questions = [t for t in tests if t['id'] in gold_queries]
print(f"Testing {len(test_questions)} questions with gold answers")
print(f"Ambiguity thresholds: H1={H1:.3f}, H2={H2:.3f}")


def clean_query(query):
    query = query.strip()
    while query and query[0] in '"\'':
        query = query[1:]
    while query and query[-1] in '"\'':
        query = query[:-1]
    return re.sub(r'\bSELECTT\b', 'SELECT', query.strip())


def result_signature(rows):
    return set(tuple(str(v) for v in row) for row in rows)

def schema_score(query: str) -> float:
    """Schema-only baseline score [7 - 0.4.2]"""
    q = query.upper()
    score = 0.0
    if "SELECT" in q: score += 0.3
    if "WHERE" in q: score += 0.3
    if "GROUP BY" in q or "ORDER BY" in q: score += 0.2
    if "FILTER" in q: score += 0.1
    if "LIMIT" in q: score += 0.1
    return score

results = []

for test in test_questions:
    qid = test['id']
    question = test['question']
    print(f"\n[{qid}] {question[:60]}")

    # Get gold results [7 - 0.4.6]
    gold_query = gold_queries[qid]
    full_gold = PREFIX + gold_query
    try:
        gold_rows = result_signature(g.query(full_gold))
    except Exception as e:
        print(f"  ❌ Gold query failed: {e}")
        continue

    # Generate candidates [7 - 0.4.1]
    try:
        cands = generate_candidates(question, schema, k=3)
        candidates = cands.get('candidates', [])
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        results.append({
            "id": qid, "correct": False,
            "baseline_correct": False, "error": str(e)
        })
        continue

    if not candidates:
        print("  ⚠️ No candidates!")
        results.append({
            "id": qid, "correct": False,
            "baseline_correct": False, "error": "no candidates"
        })
        continue

    # ── Step 1: Baseline - original order (no ML) [7 - 0.4.2] ──
    original_candidates = candidates.copy()
    baseline_query = clean_query(original_candidates[0]['query'])
    full_baseline = (PREFIX + baseline_query
                     if "PREFIX" not in baseline_query else baseline_query)
    try:
        baseline_rows = result_signature(g.query(full_baseline))
        baseline_correct = (baseline_rows == gold_rows and len(baseline_rows) > 0)
    except Exception:
        baseline_correct = False

    # ── Step 2: Feature Extraction ──
    features_list = []
    for cand in candidates:
        try:
            feats = extract_features(question, cand['query'], schema_dict)
        except Exception:
            feats = {name: 0.0 for name in FEATURE_NAMES}
        features_list.append(feats)

    X = np.array([
        [f.get(n, 0.0) for n in FEATURE_NAMES]
        for f in features_list
    ])
    X_s = scaler.transform(X)

    # ── Step 3: ML Scores (XGBoost) [7 - 0.4.5] ──
    ml_scores = model.predict_proba(X_s)[:, 1]

    # ── Step 4: Ambiguity Estimation [7][8] ──
    entropy = float(ambiguity_entropy(ml_scores))
    if entropy <= H1:
        amb_bin = "low"
    elif entropy <= H2:
        amb_bin = "mid"
    else:
        amb_bin = "high"

    # ── Step 5: Ambiguity-Gated Policy [7][8] ──
    # π(Qi) = learning-based if H1 ≤ H(Qi) ≤ H2
    # schema-based otherwise [7]
    if amb_bin == "mid":
        # Mid ambiguity → ML ranking [7][8]
        ranked = sorted(
            zip(ml_scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        policy = "xgboost"
    else:
        # Low/High ambiguity → Execution feedback [7][8]
        exec_scores = []
        for cand in candidates:
            q = clean_query(cand['query'])
            fq = PREFIX + q if "PREFIX" not in q else q
            try:
                rows = list(g.query(fq))
                exec_score = 1.0 if rows else 0.3
            except Exception:
                exec_score = 0.0
            exec_scores.append(exec_score)
        
        ranked = sorted(
            zip(exec_scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        policy = "execution-feedback"

    # ── Step 6: Top1 from gated policy ──
    ml_top_query = clean_query(ranked[0][1]['query'])
    full_ml = PREFIX + ml_top_query if "PREFIX" not in ml_top_query else ml_top_query
    try:
        ml_rows = result_signature(g.query(full_ml))
        top1_correct = ml_rows == gold_rows and len(ml_rows) > 0
    except Exception:
        top1_correct = False

        # ── Step 7: Any correct ──
    any_correct = False
    for cand in candidates:
        q = clean_query(cand['query'])
        fq = PREFIX + q if "PREFIX" not in q else q
        try:
            rows = result_signature(g.query(fq))
            if rows == gold_rows and len(rows) > 0:
                any_correct = True
                break
        except Exception:
            pass

    print(f"  Ambiguity: {amb_bin} (entropy={entropy:.3f})")
    print(f"  Policy: {policy}")
    print(f"  Baseline (no ML): {'✅' if baseline_correct else '❌'}")
    print(f"  Gated Top1:       {'✅' if top1_correct else '❌'}")
    print(f"  Any correct:      {'✅' if any_correct else '❌'}")


    results.append({
        "id": qid,
        "question": question,
        "ambiguity_bin": amb_bin,
        "entropy": entropy,
        "baseline_correct": baseline_correct,
        "top1_correct": top1_correct,
        "any_correct": any_correct,
    })

# ── Final Summary [7][8] ──
print(f"\n{'='*50}")
print(f"EVALUATION SUMMARY [7][8]")
print(f"{'='*50}")
total = len(results)
baseline = sum(1 for r in results if r.get("baseline_correct"))
top1 = sum(1 for r in results if r.get("top1_correct"))
any_c = sum(1 for r in results if r.get("any_correct"))

print(f"Total: {total}")
print(f"Baseline (no ML): {baseline}/{total} ({baseline/total*100:.1f}%)")
print(f"ML ranking:       {top1}/{total} ({top1/total*100:.1f}%)")
if baseline < top1:
    print(f"Improvement:      +{top1-baseline} (+{(top1-baseline)/total*100:.1f}%)")
elif baseline > top1:
    print(f"Degradation:      -{baseline-top1} (-{(baseline-top1)/total*100:.1f}%)")
else:
    print(f"No change")
print(f"Any correct:      {any_c}/{total} ({any_c/total*100:.1f}%)")

print(f"\nPer ambiguity level [7][8]:")
for bin_name in ["low", "mid", "high"]:
    bin_r = [r for r in results if r.get("ambiguity_bin") == bin_name]
    if bin_r:
        n = len(bin_r)
        b = sum(1 for r in bin_r if r.get("baseline_correct"))
        t = sum(1 for r in bin_r if r.get("top1_correct"))
        a = sum(1 for r in bin_r if r.get("any_correct"))
        print(f"  {bin_name}: n={n}")
        print(f"    Baseline: {b}/{n} ({b/n*100:.1f}%)")
        print(f"    ML:       {t}/{n} ({t/n*100:.1f}%)")
        print(f"    Any:      {a}/{n} ({a/n*100:.1f}%)")

# Save
with open("results/unseen_evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: results/unseen_evaluation.json")