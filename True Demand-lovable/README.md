# True Demand React UI

This frontend is the React interface for the existing True Demand KGQA runtime.
It does not query Fuseki or call the LLM directly from the browser. Requests go
through the FastAPI adapter in `../api`, which reuses the Python capability
resolver, direct templates, LLM candidate generation, ML ranking, clarification,
SPARQL execution, answer synthesis, and evidence-graph code.

## Run locally

Start Fuseki and the API from the repository root, then run:

```powershell
cd "True Demand-lovable"
npm install
npm run dev
```

The default API URL is `http://localhost:8000`. Copy `.env.example` to `.env`
only when a different backend URL is required.

The original Streamlit application remains available through `app.py`.
