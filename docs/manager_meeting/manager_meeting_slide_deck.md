# Manager Meeting Slide Deck: True Demand KGQA Thesis Progress

Use this document as the source material for PowerPoint/Google Slides. Each slide includes:

- **Slide content:** what should appear on the slide.
- **Speaker notes:** what to say verbally.
- **Visual:** figure/table/screenshot to include.

For more detailed PowerPoint speaker notes, use:

- `docs/manager_meeting/manager_meeting_expanded_speaker_notes.md`

The numbers below use the latest strict deterministic v2 audit:

- Final audited accuracy: **908/1000 = 90.8%**
- Deterministic / auto-answer route: **506/514 = 98.4%**
- LLM fallback route: **402/486 = 82.7%**
- KG analytics subset: **710/800 = 88.8%**
- DR ontology definition subset: **150/150 = 100.0%**
- Advisory subset: **48/50 = 96.0%**
- Cold-cost estimate: **486 LLM calls instead of 1000**, estimated **51.4% LLM-call reduction**
- Cached evaluation latency: direct route **142 ms avg**, LLM fallback cache route **1819 ms avg**, overall **957 ms avg**

---

## Slide 1: Title

**Slide content**

**True Demand Knowledge Graph Question Answering System**

Hybrid KGQA pipeline for:

- True Demand KG analytics
- Digital Reference ontology definitions
- Graph-grounded advisory questions
- LLM fallback with guarded ML selection

**Speaker notes**

I built a natural-language interface over the True Demand knowledge graph. The system is not only an LLM-to-SPARQL prototype. It combines deterministic graph-supported routing, ontology definition lookup, LLM candidate generation, validation, ML reranking, confidence routing, and evidence-based answer synthesis.

**Visual**

Use a clean title slide with the architecture image as a faint background or small preview:

- `docs/figures/system_architecture_pipeline.png`

---

## Slide 2: Why This Work Matters

**Slide content**

Problem:

- True Demand data is valuable but difficult to access directly.
- Business users should not need RDF, ontology, or SPARQL expertise.
- Naive LLM-to-SPARQL is risky because multiple valid-looking queries can answer different questions.

Goal:

- Make graph data accessible through natural language.
- Keep answers graph-grounded and auditable.
- Reduce LLM usage where deterministic graph paths are already known.

**Speaker notes**

The practical challenge is not only generating a query. The harder problem is selecting the query that matches the intended metric, dimension, aggregation, and scope. For example, "demand by region" can mean current demand, future demand, OEM demand, semiconductor demand, or total regional demand. The system must avoid selecting a plausible but wrong query.

**Visual**

Small conceptual graphic:

- User question -> KG/ontology -> validated answer
- Or use `docs/manager_meeting/figures/ui_entry_points.svg`

---

## Slide 3: Data Sources Used

**Slide content**

Main data artefacts:

| Artefact | Role |
|---|---|
| True Demand RDF graph | Analytical KG for survey-grounded demand, shortage, inventory, sales, region, quarter, vehicle-type data |
| True Demand schema | Compact schema abstraction used for prompting, validation, ranking, and routing |
| Digital Reference ontology | Concept-level definitions, properties, domain/range, class hierarchy |
| Benchmark datasets | 1000 final mixed questions + audit labels |

Key scale:

- True Demand graph: **>1.16M RDF triples**
- Enterprise-scale entity graph: **100k+ entities**
- Digital Reference ontology: approx. **2,742 searchable ontology terms**

**Speaker notes**

The True Demand KG is the analytical backend. It contains survey-grounded data and RDF triples. The Digital Reference ontology is different: it provides model-level meaning and definitions. I added it so users can ask "What is a Technology Node?" or "What does is processed by mean?" without using the LLM.

**Visual**

Use a two-layer diagram:

- True Demand KG = data layer
- DR ontology = conceptual/definition layer

Optional existing figures:

- `docs/figures/webvowl_ontology_view.png`
- `docs/figures/true_demand_ontology_relationships.png`

---

## Slide 4: What the Graph Can Answer

**Slide content**

The system covers three question families:

1. **KG analytics**
   - future demand by region / quarter / vehicle type / technology category
   - current demand by region / survey group / BL1-BL2 baseline
   - shortages by survey type / company / shortage status
   - vehicle sales by month / year / actual vs forecast
   - autonomous driving percentages by vehicle type / SAE level / year
   - inventory by component / technology category / trend

2. **Ontology definitions**
   - "What is a Technology Node?"
   - "Define Demand Class."
   - "What does has customer part number specification mean?"

3. **Advisory questions**
   - "What should I inspect first to understand shortage exposure?"
   - "Which area should be monitored more closely?"

**Speaker notes**

I separated analytical questions from ontology questions. This is important because "What is Future Demand?" should not become a count over FutureDemandAnalysis entries. It should be routed to deterministic ontology lookup.

**Visual**

Use three columns: KG analytics / DR definitions / Advisory.

---

## Slide 5: User Interface Entry Points

**Slide content**

The UI supports multiple user paths:

- Free-text natural-language question
- Validated examples
- Guided question builder
- Available topics / capabilities
- Digital Reference ontology browser
- Evidence graph after answer

**Speaker notes**

The user is not forced to know the exact schema. They can type freely, choose a validated example, build a supported graph query through the builder, or browse ontology terms. The key design principle is that suggested questions should be graph-supported, while free text can still go through routing and fallback.

**Visual**

Use:

- `docs/manager_meeting/figures/ui_entry_points.svg`

Optional screenshot:

- `docs/figures/ui_answer_page.png`

---

## Slide 6: Final System Architecture

**Slide content**

Hybrid KGQA pipeline:

1. Streamlit UI
2. Request router
3. Deterministic route if exact graph/ontology route exists
4. LLM fallback only if unsupported or ambiguous
5. Validation + feature extraction
6. Guarded ML selection
7. Confidence routing
8. Execution, answer synthesis, evidence graph, audit logs

**Speaker notes**

The architecture is designed to avoid unnecessary LLM calls. Deterministic routes are used first when the metric, dimension, aggregation, and scope are confidently supported. If no exact route exists, the system generates SPARQL candidates with the LLM, validates them, reranks them, and either answers or asks for clarification.

**Visual**

Use:

- `docs/figures/system_architecture_pipeline.png`

---

## Slide 7: Deterministic Routing

**Slide content**

Deterministic routing is used when:

- capability is known
- metric is known
- dimension is supported
- aggregation intent is clear
- query path exists in the graph
- execution returns graph evidence

Examples:

- "Show future demand by region."
- "How many OEM companies reported shortages?"
- "What is a Technology Node?"
- "Show monthly vehicle sales, actual vs forecast."

**Speaker notes**

The deterministic route is not a keyword shortcut. It is a controlled capability inventory. A question is answered directly only if the system has an explicit supported path and the requested shape makes sense. This is why the deterministic route is now high accuracy.

**Visual**

Use:

- `docs/figures/direct_graph_supported_routing.png`

---

## Slide 8: When the System Falls Back to LLM

**Slide content**

LLM fallback is used when:

- no exact deterministic graph path exists
- the question is genuinely ambiguous
- multiple interpretations remain possible
- metric/dimension/scope cannot be resolved safely
- the deterministic candidate would be empty or semantically unsafe

Examples:

- complex autonomous-driving grouping by vehicle type, SAE level, year, and survey provenance
- future demand across multiple dimensions
- current demand with BL1/BL2 and market segment constraints
- graph paths where the schema is not directly aligned

**Speaker notes**

Fallback does not mean failure. It means the question is outside the safe deterministic boundary. These are usually the cases where even a human needs to inspect the schema and test SPARQL paths. The LLM provides candidate queries, but the system still validates and ranks them.

**Visual**

Use the right half of:

- `docs/figures/system_architecture_pipeline.png`

Or create a simple route decision slide.

---

## Slide 9: Candidate Generation vs Selection

**Slide content**

Key distinction:

- **Generation quality:** whether the correct query exists among candidates.
- **Selection quality:** whether the system chooses the correct candidate.
- **Answer-level correctness:** whether the final user-facing answer is correct.

Why this matters:

- A candidate set can contain the right query but still select the wrong one.
- Several different queries can sometimes produce the same correct answer.
- Final system accuracy must be audited at answer level.

**Speaker notes**

This was one of the most important research findings. The LLM is often good at generating a correct candidate somewhere in the list, but ranking and selecting the correct one is the bottleneck. That is why the architecture separates generation, validation, selection, and final answer correctness.

**Visual**

Use a three-stage diagram:

Question -> top-k candidates -> selected query -> final answer

Optional:

- `docs/figures/confidence_ambiguity_routing.png`

---

## Slide 10: Confidence and Ambiguity Routing

**Slide content**

The system uses:

- candidate scores
- score margin
- normalized entropy / uncertainty
- schema and shape validation
- execution evidence
- safety flags

Decision policy:

- high confidence -> answer
- medium ambiguity -> ML/semantic reranking
- high ambiguity -> clarification or controlled no-answer

**Speaker notes**

The confidence routing implements the thesis idea: ambiguity is not uniform. Low-ambiguity cases can be answered directly or with high confidence. Medium cases benefit from ML reranking. High-ambiguity cases should not be forced into an answer if evidence is weak.

**Visual**

Use:

- `docs/figures/confidence_ambiguity_routing.png`

---

## Slide 11: Final Benchmark Design

**Slide content**

Final benchmark:

| Question family | Count | Purpose |
|---|---:|---|
| KG analytics | 800 | True Demand analytical questions |
| DR ontology definitions | 150 | deterministic ontology-model lookup |
| Advisory questions | 50 | graph-grounded conservative recommendations |
| **Total** | **1000** | end-to-end user-facing system evaluation |

**Speaker notes**

The final benchmark is mixed intentionally. A real user does not only ask numerical queries. They also ask what concepts mean, what fields represent, and what should be inspected first. This is why the system-level benchmark includes KG analytics, ontology definitions, and advisory questions.

**Visual**

Use:

- `docs/manager_meeting/figures/benchmark_composition.svg`

---

## Slide 12: Evaluation Methodology

**Slide content**

How I evaluated correctness:

1. Ran all 1000 questions through the full system.
2. Logged route, source, selected query, graph rows, answer text, and evidence.
3. Built an audit CSV.
4. Manually labeled each answer as correct or incorrect.
5. Added notes and failure categories.
6. Classified incorrect cases by human SPARQL difficulty:
   - easy
   - medium
   - hard

**Speaker notes**

The final accuracy is not based only on automatic metrics. I manually audited the final answers. For every row I checked whether the final answer matched the question intent, the requested metric, grouping, scope, and graph evidence. I also separated simple avoidable failures from hard schema-dependent failures.

**Visual**

Use a screenshot/table excerpt from:

- `results/kgqa_system_accuracy_audit_1000_strict_direct_v2_labeled_codex.csv`

Show columns:

- question
- route
- selected source
- selected query
- answer
- correctness
- human SPARQL difficulty
- failure family

---

## Slide 13: Final Accuracy Results

**Slide content**

| Route | Questions | Correct | Accuracy |
|---|---:|---:|---:|
| Deterministic / auto-answer | 514 | 506 | **98.4%** |
| LLM fallback | 486 | 402 | **82.7%** |
| **Overall system** | **1000** | **908** | **90.8%** |

**Speaker notes**

The strict deterministic route achieved 98.4% accuracy, which means the controlled graph-supported path is reliable when it fires. The LLM fallback is lower because it handles the unresolved and more ambiguous cases. Overall, the system reaches 90.8% audited answer-level accuracy on the full 1000-question benchmark.

**Visual**

Use:

- `docs/manager_meeting/figures/route_accuracy.svg`

---

## Slide 14: Accuracy by Question Family

**Slide content**

| Question family | Questions | Correct | Accuracy |
|---|---:|---:|---:|
| KG analytics | 800 | 710 | **88.8%** |
| DR ontology definitions | 150 | 150 | **100.0%** |
| Advisory questions | 50 | 48 | **96.0%** |

**Speaker notes**

The DR ontology route is deterministic and reached 100% on the benchmark. Advisory questions reached 96%, because most advisory templates are row-backed and conservative. KG analytics is the hardest subset because it requires metric, dimension, aggregation, and graph-path alignment.

**Visual**

Use simple table or bar chart derived from these numbers.

---

## Slide 15: Correct vs Incorrect by Route

**Slide content**

Route-level answer correctness:

- Deterministic: **506 correct**, **8 incorrect**
- LLM fallback: **402 correct**, **84 incorrect**

Interpretation:

- deterministic route is precise after stricter guards
- remaining LLM errors concentrate in complex schema-dependent families

**Speaker notes**

This is the closest useful version of a true/false matrix for this system. It is not a traditional classifier confusion matrix, because the system always produces a route and an answer or fallback. But it clearly shows where incorrect answers are concentrated.

**Visual**

Use:

- `docs/manager_meeting/figures/route_correctness_heatmap.svg`

---

## Slide 16: Cost and Latency Impact

**Slide content**

Cold-cost estimate:

| Mode | LLM calls | Cost at EUR 0.20/call |
|---|---:|---:|
| All-LLM baseline | 1000 | EUR 200.00 |
| Hybrid system | 486 | EUR 97.20 |
| Savings | 514 skipped | EUR 102.80 |

Estimated LLM-call reduction: **51.4%**

Observed cached latency:

- deterministic direct: **142 ms avg**
- cached LLM + ranking: **1819 ms avg**
- overall: **957 ms avg**

**Speaker notes**

The evaluation run used cached LLM candidates, so the actual log reports zero fresh LLM calls. For a fair cold-run estimate, I count the 486 fallback cases as LLM calls. Compared with sending all 1000 questions to the LLM, the hybrid system skips 514 calls and reduces estimated LLM cost by 51.4%.

**Visual**

Use:

- `docs/manager_meeting/figures/cost_latency.svg`

---

## Slide 17: Why Remaining Errors Happen

**Slide content**

Remaining incorrect answers: **92**

By human SPARQL difficulty:

| Difficulty | Incorrect cases |
|---|---:|
| Easy | 8 |
| Medium | 21 |
| Hard | 63 |

Main interpretation:

- Only a small number are easy avoidable failures.
- Most failures are hard schema-dependent query families.

**Speaker notes**

I classified the incorrect cases by asking: if I were a human with schema access, how hard would it be to write the correct SPARQL? Only 8 cases are easy. The majority require schema inspection, indirect joins, or careful composition across survey scope, dimensions, and metrics.

**Visual**

Use:

- `docs/manager_meeting/figures/error_difficulty.svg`

---

## Slide 18: Error Family Breakdown

**Slide content**

Top remaining error families:

| Failure family | Incorrect |
|---|---:|
| Autonomous driving complex grouping | 24 |
| Current demand BL/scope | 20 |
| Vehicle sales metric/dimension | 14 |
| Future demand complex dimension | 12 |
| Other semantic mismatch | 9 |

**Speaker notes**

The failures are not random. They are concentrated in a small number of graph-pattern families. This is useful because it tells us where future improvement should focus: not generic prompt tuning, but better graph paths, explicit templates, and schema alignment for these complex families.

**Visual**

Use:

- `docs/manager_meeting/figures/failure_families.svg`

---

## Slide 19: Examples of Correct Deterministic Answers

**Slide content**

Examples:

1. "What is a Technology Node?"
   - route: DR ontology lookup
   - no LLM call

2. "Show future demand by region."
   - route: KG direct template
   - grouped breakdown

3. "How many OEM companies reported shortages?"
   - route: shortage template
   - scoped count over graph rows

4. "Show monthly vehicle sales actual vs forecast."
   - route: vehicle-sales template
   - grouped by month and sales type

**Speaker notes**

These examples show the benefit of deterministic routing. The system knows the graph path and does not rely on the LLM. This improves reliability, speed, and cost.

**Visual**

Use 2 screenshots from the UI showing:

- DR definition answer
- KG analytic answer with evidence graph

If screenshots are not ready, use a simple table with question, route, and answer type.

---

## Slide 20: Examples of Hard LLM Fallback Cases

**Slide content**

Examples of difficult cases:

- autonomous driving by vehicle type + SAE level + year + survey provenance
- current demand BL1/BL2 filtered to Tier1 Automotive
- future demand by quarter and technology category
- regional demand combined with vehicle type

Why difficult:

- multiple valid-looking graph paths
- survey scope must be preserved
- some dimensions are not directly connected
- non-empty query can still be semantically wrong

**Speaker notes**

These are the questions where the LLM is still useful but also risky. It can generate candidate paths that would be hard to write manually, but selection remains difficult because several candidates can be structurally valid and non-empty.

**Visual**

Use a failure example row from the audit CSV, showing:

- question
- selected query
- why wrong
- human difficulty = hard

---

## Slide 21: Digital Reference Integration

**Slide content**

DR ontology support:

- deterministic ontology lookup
- searchable ontology browser
- definitions and labels
- domain/range for properties
- subclass / related-term navigation
- no LLM needed for definition questions

Benchmark result:

- **150/150 = 100%** on DR definition questions

**Speaker notes**

The DR ontology strengthens the system by adding conceptual understanding. It lets the user ask what a concept or property means, separate from analytical KG questions. This is important for non-technical users because it helps them understand the model before asking analytical questions.

**Visual**

Use screenshot of Digital Reference tab / ontology browser, or:

- `docs/figures/webvowl_ontology_view.png`

---

## Slide 22: Evidence and Auditability

**Slide content**

Every answer logs:

- question
- route
- selected source
- selected query
- graph row count
- row preview
- answer text
- execution reason
- correctness audit label
- failure type and difficulty when wrong

**Speaker notes**

This is important for trust. The system does not only display an answer. It records how the answer was produced and what graph evidence was used. This makes the evaluation auditable and also helps debug failures.

**Visual**

Use CSV screenshot or a compact audit-flow diagram.

---

## Slide 23: What Has Improved Recently

**Slide content**

Recent improvements:

- stricter deterministic routing guards
- better distinction between ontology definitions and KG analytics
- broader direct templates for common graph-supported questions
- shortage scope handling fixes
- actual-vs-forecast vehicle sales routing
- rank/top query handling
- validated examples and guided builder improvements
- interactive evidence graph improvements

**Speaker notes**

The most important change was making deterministic routing stricter. The system should answer directly only when it can do so correctly. Otherwise, it should fall back to the LLM. After these changes, deterministic accuracy reached 98.4%.

**Visual**

Before/after metric callout:

- previous strict working estimate was not final
- final strict v2: deterministic 98.4%, overall 90.8%

---

## Slide 24: What I Would Improve Next

**Slide content**

Next steps:

- improve remaining hard query families
- add more graph-supported deterministic paths where schema supports them
- improve answer synthesis for grouped and comparison results
- enhance evidence graph interaction and readability
- add execution-signature clustering for candidate equivalence
- continue manual audit on new benchmark samples

**Speaker notes**

The next improvements should not be random prompt tuning. The error analysis shows where the bottlenecks are. The best path is targeted: better graph modeling or templates for the recurring hard families, plus improved ambiguity handling when multiple interpretations are still viable.

**Visual**

Roadmap timeline: short-term / medium-term / thesis final.

---

## Slide 25: Closing Message

**Slide content**

Main conclusion:

The system makes the True Demand KG accessible to non-technical users through a hybrid KGQA architecture.

Final audited results:

- **90.8%** answer-level accuracy on 1000 mixed questions
- **98.4%** deterministic-route accuracy
- **82.7%** LLM fallback accuracy
- **100.0%** DR ontology definition accuracy
- **51.4%** estimated LLM-call reduction

**Speaker notes**

The strongest result is not only the accuracy number. It is the architecture: deterministic where possible, LLM only where needed, ML/validation for selection, and manual audit to measure the final user-facing answer honestly.

**Visual**

Use:

- `docs/manager_meeting/figures/route_accuracy.svg`
- or a final KPI card slide.

---

# Appendix Slides

## Appendix A: Exact Evaluation Files

Use these as material if the manager asks where the numbers come from:

- Final audit CSV:
  - `results/kgqa_system_accuracy_audit_1000_strict_direct_v2_labeled_codex.csv`
- Final audit summary:
  - `results/kgqa_system_accuracy_audit_1000_strict_direct_v2_labeled_codex_summary.md`
- Raw full-system JSONL:
  - `/Users/elenetsaouse/Downloads/kgqa_system_accuracy_1000_strict_direct_v2_full.jsonl`
- Efficiency report:
  - `/Users/elenetsaouse/Downloads/kgqa_efficiency_1000_strict_direct_v2_full.json`

## Appendix B: How to Explain 90.8% Accuracy

Say:

"I evaluated 1000 mixed user-facing questions. For each question, I logged the route, selected query, graph rows, and answer. I then manually audited whether the final answer matched the user intent. The final audited result was 908 correct out of 1000, or 90.8%."

Do not say:

"The model accuracy is 90.8%."

More precise wording:

"The complete KGQA system achieved 90.8% audited answer-level accuracy on the final mixed benchmark."

## Appendix C: How to Explain Cost

Say:

"The actual evaluation run used cached LLM outputs, so the log reports zero fresh LLM calls. For a fair cold-run estimate, I count the 486 fallback questions as LLM calls. Compared with 1000 calls in an all-LLM baseline, this saves 514 calls, or 51.4%."

## Appendix D: How to Explain Remaining Errors

Say:

"The remaining 92 errors are not evenly distributed. Most are in a small number of hard graph families: autonomous-driving complex grouping, current-demand baseline and scope, vehicle-sales metric/dimension, and future-demand multi-dimensional queries. Only 8 errors were easy cases that a schema-aware human would likely formulate quickly."
