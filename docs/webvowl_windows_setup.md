# WebVOWL Ontology Viewer Setup

This project can use WebVOWL as the ontology visualization framework for the
True Demand KG. Fuseki remains the SPARQL execution backend; WebVOWL is only for
ontology/schema visualization.

## What Each Tool Does

- **Apache Jena Fuseki** executes SPARQL over the full RDF graph.
- **OWL2VOWL** converts a TTL/OWL ontology file into WebVOWL JSON.
- **WebVOWL** renders the ontology graph in the browser.

## 1. Start WebVOWL

Use the official WebVOWL project. The simplest route is Docker:

```powershell
git clone https://github.com/VisualDataWeb/WebVOWL.git
cd WebVOWL
docker-compose up -d
```

Open:

```text
http://localhost:8080
```

## 2. Get OWL2VOWL

Download or build `owl2vowl.jar` from:

```text
https://github.com/VisualDataWeb/OWL2VOWL
```

The converter requires Java. Test it with:

```powershell
java -jar C:\path\to\owl2vowl.jar -file C:\path\to\ontology.ttl -echo
```

## 3. Configure the Streamlit App

In Developer settings:

```text
Ontology path for WebVOWL = data\infineon\ontology.ttl
WebVOWL app URL = http://localhost:8080
OWL2VOWL jar path = C:\path\to\owl2vowl.jar
```

Or set these in `.env`:

```env
TRUE_DEMAND_ONTOLOGY_PATH=C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\data\infineon\ontology.ttl
WEBVOWL_URL=http://localhost:8080
OWL2VOWL_JAR_PATH=C:\path\to\owl2vowl.jar
```

Then run:

```powershell
streamlit run app.py
```

## 4. Using It

The app converts the ontology or answer-evidence slice to WebVOWL JSON and shows
a download button. Load that JSON in the embedded/local WebVOWL panel.

Use `ontology.ttl` for the main ontology view. Do not send the full
`graph.ttl` to WebVOWL unless you intentionally want a very large and noisy
instance-level graph.

