# Fuseki Windows Setup

This setup keeps the UI unchanged and moves SPARQL execution from local RDFLib
to a local Apache Jena Fuseki endpoint.

## 1. Install prerequisites

- Java 17 or newer.
- Apache Jena Fuseki distribution zip from the official Apache Jena downloads.

Unzip Fuseki somewhere simple, for example:

```powershell
C:\tools\apache-jena-fuseki
```

## 2. Start Fuseki with the Infineon graph

From the unzipped Fuseki directory:

```powershell
.\fuseki-server.bat --file=C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\data\infineon\graph.ttl /infineon
```

The SPARQL endpoint is:

```text
http://localhost:3030/infineon/sparql
```

The browser query UI is:

```text
http://localhost:3030/#/dataset/infineon/query
```

Keep this PowerShell window open while using the KGQA app.

## 3. Point the KGQA app to Fuseki

In `.env`, add:

```env
FUSEKI_QUERY_URL=http://localhost:3030/infineon/sparql
```

Then start Streamlit as usual:

```powershell
streamlit run app.py
```

If developer mode is enabled in the app, you can also set the endpoint in:

```text
Developer settings -> Data -> Fuseki query endpoint
```

## 4. Quick endpoint test

Open the Fuseki browser query UI and run:

```sparql
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>

SELECT (COUNT(*) AS ?count)
WHERE {
  ?s ?p ?o .
}
```

If this returns a count, Fuseki loaded the graph.

## Notes

- Local Fuseki does not need an Infineon token.
- The Infineon GPT token is still needed for LLM candidate generation.
- `FUSEKI_QUERY_URL` affects SPARQL execution only; ranking, confidence routing,
  clarification, and the UI stay the same.
- If `FUSEKI_QUERY_URL` is empty, the app falls back to local `graph.ttl` with RDFLib.
