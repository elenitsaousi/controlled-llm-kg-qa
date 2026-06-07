# Family-Aware Schema Routing

Family-aware schema routing reduces the ontology/schema text sent to the LLM
before candidate generation. It does not split the graph and does not change
SPARQL execution.

Runtime flow:

```text
Question
-> cheap family router
-> focused schema prompt, or full schema if router confidence is low
-> LLM candidate generation
-> ML reranking and confidence routing
-> full Fuseki/RDFLib graph execution
```

## Enable in Streamlit

Open developer mode and use:

```text
Developer settings -> Schema Routing -> Family-aware schema routing
```

Recommended first-test settings:

```text
Family-aware schema routing: on
Max routed families: 3
Retry full schema after sliced prompt: off
```

Keeping full-schema retry off avoids a second LLM call. If routing confidence is
low, the system uses the full schema from the start.

## Enable through `.env`

```env
INFINEON_ENABLE_SCHEMA_SLICING=1
INFINEON_SCHEMA_SLICING_MAX_FAMILIES=3
INFINEON_SCHEMA_SLICING_FULL_FALLBACK=0
```

To return to the previous behavior:

```env
INFINEON_ENABLE_SCHEMA_SLICING=0
```

or disable the checkbox in the UI.

## Families

Current families:

- `vehicle_sales`
- `future_demand`
- `regional_demand`
- `current_demand_baselines`
- `inventory`
- `shortage`
- `order_cancellation`
- `autonomous_driving`
- `catalog_lookup`

The router can select multiple families for combined questions, for example
current demand by region.
