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

The repository also includes a precomputed WebVOWL export for the current True
Demand ontology:

```text
data/infineon/true_demand_webvowl.json
```

After `git pull`, you can upload this file directly in WebVOWL. You do not need
to run OWL2VOWL on the second machine for the main ontology view.

To open WebVOWL locally without Docker after building the deploy folder:

```powershell
cd C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\WebVOWL
python -m http.server 8080 -d deploy
```

Then open:

```text
http://localhost:8080
```

In WebVOWL, load:

```text
C:\Users\tsaousieleni\Documents\controlled-llm-kg-qa\data\infineon\true_demand_webvowl.json
```

In the Streamlit app, `Graph Overview -> WebVOWL Ontology Viewer` embeds the
local WebVOWL page and exposes the same JSON as a download. WebVOWL itself still
controls ontology loading; if it opens with an example ontology, load the True
Demand JSON from the WebVOWL ontology/upload menu.

Use `ontology.ttl` for the main ontology view. Do not send the full
`graph.ttl` to WebVOWL unless you intentionally want a very large and noisy
instance-level graph.
