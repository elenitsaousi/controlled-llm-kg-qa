# True Demand KGQA Overview Assets

This folder contains slide-ready overview material for explaining the thesis work.

## Images

- `system_architecture.svg`
  Shows the current end-to-end True Demand KGQA architecture: Streamlit UI, request preparation, optional family-aware schema routing, Infineon GPT candidate generation, ML selection, Fuseki execution, confidence routing, clarification, logs, and feedback.

- `true_demand_ontology_relationships.svg`
  Shows the main True Demand ontology areas used by the system: surveys, companies, demand analyses, regions, inventory, order cancellation, vehicle sales, and autonomous driving.

- `definition_fallback_flow.svg`
  Shows the planned handling for definition questions: first True Demand ontology, then Digital Reference exact match, then close match, then curated fallback definitions.

- `evaluation_loop.svg`
  Shows the current evaluation story: 1000-question benchmark, gold validation, held-out evaluation, feature iteration, forced Top-1 results, candidate recall, and confidence-aware routing.

## Suggested Presentation Order

1. Goal and scope: natural-language Q&A over the True Demand KG.
2. True Demand ontology: key classes and relationships.
3. System architecture: what goes into the LLM and what is handled deterministically.
4. Evaluation: benchmark, validation, metrics, error analysis.
5. Current limitations: accuracy, candidate generation, ambiguity, noisy graph data, API token stability.
6. Next steps: latency measurement, selection failure reduction, feedback-driven improvements, and simple production-style routing.

## Related Docs

- `overview/slide_outline.md`: short explanation for a supervisor/status update.
- `docs/project_artifact_cleanup.md`: conservative cleanup guide for generated results and model artifacts.
