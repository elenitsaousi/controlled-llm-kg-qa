# Infineon Data Splits

Use these canonical files for the current branch:

- `infineon_train.json`: 300 examples used to train the query-plan predictor and any ML ranker.
- `infineon_dev.json`: 100 examples used for prompt, template, ranking, and execution-selection tuning.
- `infineon_test_final.json`: 50 examples reserved for final evaluation. Do not tune code against this file; after inspecting its failures, promote it to dev and create a new final test.

The graph and schema inputs remain:

- `graph.ttl`
- `schema.json`
- `ontology.ttl`

Older generated datasets and previous evaluation sets live in `archive/` so the active data directory stays small and the split roles are explicit.

## Ontology Semantics

The graph uses the following distinction:

- `owl:Class` / `rdfs:Class`: domain concepts such as `Region`, `Survey`, and `VehicleType`.
- `owl:ObjectProperty`: relationships whose objects are RDF resources, such as `hasSurveyOrigin` and `inRegion`.
- `owl:DatatypeProperty`: attributes whose objects are literals, such as `companyName`, `hasYear`, and `isActiveInCategory`.
- Individuals: concrete observations, companies, quarters, survey responses, and category values typed with `rdf:type`.

Validate the complete graph after every data-generation or ontology-cleanup change:

```powershell
python validation\validate_ontology_semantics.py --graph data\infineon\graph.ttl
```

The validator rejects unclassified properties, mixed literal/resource usage,
and property declarations that disagree with their actual objects. Class and
individual punning is reported separately as a warning because it can be valid
OWL modeling but should be reviewed deliberately.
