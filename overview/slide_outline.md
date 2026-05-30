# Short Slide Outline

## 1. Goal

Build an LLM-based question answering system over the Infineon True Demand knowledge graph, so users can ask business questions without writing SPARQL or knowing the ontology structure.

## 2. What Exists

- True Demand ontology and RDF graph
- Business concepts such as surveys, companies, regions, demand, inventory, vehicle sales, autonomous driving, and technology categories
- Streamlit prototype for asking questions and inspecting results

## 3. How It Works

The LLM does not answer directly from memory. It receives controlled ontology/schema context and generates SPARQL candidates. The system then validates, executes, ranks, and explains the selected query.

## 4. What Is Stored

- RDF graph
- Ontology/schema description
- Entity alias/profile information
- Benchmark questions and gold SPARQL queries
- ML reranker model
- Evaluation results and error analysis

## 5. What I Implemented

- End-to-end KGQA pipeline
- Streamlit UI with graph overview and explanations
- 360-question benchmark
- Gold query validation
- Duplicate and wording audits
- Held-out evaluation and error analysis
- Initial robustness layer for noisy entities using structural profiles
- Infineon LLM token auth checks and token-refresh support

## 6. What I Learned

The main bottleneck is not only ranking. In many cases, the correct SPARQL query is not generated at all, so candidate generation and schema/entity grounding are the next priority.

## 7. Next Steps

- Use a smaller task-specific subgraph for faster and more reliable answers
- Improve candidate generation for weak question families
- Improve ambiguity handling
- Add definition fallback using Digital Reference exact/close matches
- Stabilize API/token access for long evaluations

