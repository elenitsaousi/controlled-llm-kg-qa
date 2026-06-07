# True Demand KGQA Overview Assets

This folder contains slide-ready overview material for explaining the thesis work.

## Images

- `system_architecture.svg`
  Shows the end-to-end KGQA pipeline: user question, entity linking, ontology context, LLM candidate generation, validation, ranking, execution, answer, and explanation.

- `true_demand_ontology_relationships.svg`
  Shows the main True Demand ontology areas used by the system: surveys, companies, demand analyses, regions, inventory, order cancellation, vehicle sales, and autonomous driving.

- `definition_fallback_flow.svg`
  Shows the planned handling for definition questions: first True Demand ontology, then Digital Reference exact match, then close match, then curated fallback definitions.

- `evaluation_loop.svg`
  Shows how the benchmark/evaluation loop works: benchmark, gold SPARQL validation, generated candidates, execution, metrics, error analysis, and targeted improvements.

## Suggested Presentation Order

1. Goal and scope: natural-language Q&A over the True Demand knowledge graph.
2. True Demand ontology: key classes and relationships.
3. System architecture: what goes into the LLM and what is handled deterministically.
4. Evaluation: benchmark, validation, metrics, error analysis.
5. Current limitations: accuracy, candidate generation, ambiguity, noisy graph data, API token stability.
6. Next steps: smaller task-specific subgraph, definition fallback with Digital Reference, stronger generation and ambiguity handling.
