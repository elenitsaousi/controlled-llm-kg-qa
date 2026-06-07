# Short Slide Outline

## 1. Goal

Build a True Demand KGQA assistant that lets business users ask natural-language questions over the True Demand knowledge graph without writing SPARQL or knowing the ontology structure.

## 2. Current System

- Streamlit UI for free-text questions, examples, graph overview, confidence dashboard, and interactive graph exploration.
- Infineon GPT generates multiple SPARQL candidates from controlled ontology/schema context.
- Apache Jena Fuseki is the default graph execution backend.
- A feature-based ML selector ranks generated SPARQL candidates.
- Confidence routing decides whether to answer directly or ask the user to choose among natural-language interpretations.

## 3. Architecture

The LLM does not answer directly from memory. It proposes candidate SPARQL queries. The system then validates, ranks, executes, and explains the selected query using the True Demand KG.

Main stages:

1. User question
2. Request preparation and optional family-aware schema routing
3. LLM candidate generation
4. Candidate validation and ML ranking
5. Confidence-aware routing
6. Fuseki SPARQL execution
7. Graph-grounded answer, explanation, feedback, and logs

## 4. What Is Stored

- True Demand RDF graph
- Ontology/schema description
- Benchmark questions and gold SPARQL queries
- ML reranker model
- Evaluation reports and error analyses
- Session and feedback logs from the UI

## 5. What I Implemented

- End-to-end KGQA pipeline from question to SPARQL answer
- 1000-question benchmark plan and generated benchmark dataset
- Gold-query validation, wording audits, duplicate checks, and split creation
- Held-out evaluation, error analysis, switch audits, and high-confidence mistake analysis
- Feature-based selection/ranking improvements using answer-shape, scope, origin, grouping, and output-variable signals
- XGBoost classifier and XGBoost learning-to-rank experiments
- Confidence-aware routing and top-3 clarification options in natural language
- Streamlit UI with True Demand KG QA, confidence dashboard, graph overview, interactive graph explorer, Fuseki backend, feedback logging, and timing breakdown
- Infineon LLM auth checks and optional token refresh support

## 6. Key Results So Far

- Candidate recall / Any-Correct on the final 1000-question test split: about **94.4%**.
- Forced Top-1 selection: about **67-69%**, depending on selection/routing configuration.
- Selective answering with high confidence: about **90.2% accuracy at 30.8% coverage**.
- Main bottleneck: the correct query is often generated but not always selected first.
- Practical implication: the system should not always force an answer; it should answer high-confidence cases and ask for clarification otherwise.

## 7. Current Limitations

- Some selection failures remain, especially where top candidates are semantically close.
- Latency depends strongly on LLM response time and graph/query execution.
- Some ontology nodes have little human-readable descriptive metadata.
- Clarification quality depends on whether the candidate queries can be summarized clearly.

## 8. Next Steps

- Measure latency breakdown systematically over a small question set.
- Improve remaining high-confidence mistakes and selection failures.
- Use feedback logs to identify common user-facing failure patterns.
- Keep Fuseki as the default execution backend.
- Keep the UI simple: answer, clarification when needed, graph exploration only on demand.
