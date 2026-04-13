import json
from collections import defaultdict
from pathlib import Path

from rdflib import Graph, RDF, RDFS, OWL, URIRef, Literal


BASES = {
    "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/",
    "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey#",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_TTL = PROJECT_ROOT / "data" / "infineon" / "ontology.ttl"
GRAPH_TTL = PROJECT_ROOT / "data" / "infineon" / "graph.ttl"
OUT_SCHEMA = PROJECT_ROOT / "data" / "infineon" / "schema.json"


def _local_name(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#", 1)[1]
    return s.rsplit("/", 1)[-1]


def _in_base(uri: URIRef) -> bool:
    uri_str = str(uri)
    return any(uri_str.startswith(base) for base in BASES)


def _add_relationship(rel_map, pred_local, from_types, to_types):
    if pred_local not in rel_map:
        rel_map[pred_local] = {"from": set(), "to": set()}
    rel_map[pred_local]["from"].update(from_types)
    rel_map[pred_local]["to"].update(to_types)


def _load_ontology():
    g = Graph()
    g.parse(str(ONTOLOGY_TTL), format="turtle")
    return g


def _load_graph():
    g = Graph()
    g.parse(str(GRAPH_TTL), format="turtle")
    return g


def build_schema():
    classes = set()
    predicates = set()
    properties = set()
    relationships = {}

    # --- From ontology (if defined) ---
    g_onto = _load_ontology()
    for s in g_onto.subjects(RDF.type, OWL.Class):
        if _in_base(s):
            classes.add(_local_name(s))
    for s in g_onto.subjects(RDF.type, RDFS.Class):
        if _in_base(s):
            classes.add(_local_name(s))

    for s in g_onto.subjects(RDF.type, OWL.ObjectProperty):
        if _in_base(s):
            predicates.add(_local_name(s))
    for s in g_onto.subjects(RDF.type, OWL.DatatypeProperty):
        if _in_base(s):
            properties.add(_local_name(s))
    for s in g_onto.subjects(RDF.type, RDF.Property):
        if _in_base(s):
            predicates.add(_local_name(s))

    # Use explicit domain/range if present
    rel_map = {}
    for prop in list(predicates) + list(properties):
        domains = set()
        ranges = set()
        for base in BASES:
            prop_uri = URIRef(base + prop)
            domains.update(
                _local_name(o)
                for o in g_onto.objects(prop_uri, RDFS.domain)
                if _in_base(o)
            )
            ranges.update(
                _local_name(o)
                for o in g_onto.objects(prop_uri, RDFS.range)
                if _in_base(o)
            )
        if domains or ranges:
            rel_map[prop] = {"from": set(domains), "to": set(ranges)}

    # Add subclass relations from ontology if present
    for s, _, o in g_onto.triples((None, RDFS.subClassOf, None)):
        if _in_base(s) and _in_base(o):
            sub = _local_name(s)
            sup = _local_name(o)
            classes.add(sub)
            classes.add(sup)
            _add_relationship(rel_map, "subClassOf", {sub}, {sup})

    # --- Enrich from graph if ontology is sparse ---
    if len(classes) < 10 or (len(predicates) + len(properties)) < 10:
        g = _load_graph()

        # Map node -> rdf:types (only survey namespace)
        type_map = defaultdict(set)
        for s, _, o in g.triples((None, RDF.type, None)):
            if _in_base(o):
                type_map[s].add(_local_name(o))
                classes.add(_local_name(o))
            elif o == RDFS.Class and _in_base(s):
                # Some classes are declared as rdfs:Class in graph.ttl
                local_s = _local_name(s)
                classes.add(local_s)
                type_map[s].add(local_s)

        # Collect predicates/properties and relationships from observed data
        for s, p, o in g:
            if p == RDFS.subClassOf and _in_base(s) and _in_base(o):
                sub = _local_name(s)
                sup = _local_name(o)
                classes.add(sub)
                classes.add(sup)
                _add_relationship(rel_map, "subClassOf", {sub}, {sup})
                continue

            if not _in_base(p):
                continue

            pred_local = _local_name(p)
            if isinstance(o, Literal):
                properties.add(pred_local)
            else:
                predicates.add(pred_local)

            from_types = set(type_map.get(s, set()))
            if not from_types and _in_base(s):
                local_s = _local_name(s)
                if local_s in classes:
                    from_types = {local_s}

            to_types = set()
            if not isinstance(o, Literal):
                to_types = set(type_map.get(o, set()))
                if not to_types and _in_base(o):
                    local_o = _local_name(o)
                    if local_o in classes:
                        to_types = {local_o}
            if from_types or to_types:
                _add_relationship(rel_map, pred_local, from_types, to_types)

    # Finalize relationships
    relationships = []
    for pred, rel in sorted(rel_map.items()):
        relationships.append(
            {
                "type": pred,
                "from": sorted(rel["from"]),
                "to": sorted(rel["to"]),
            }
        )

    schema = {
        "description": (
            "Schema extracted from ontology.ttl and observed graph.ttl predicates/classes."
        ),
        "classes": sorted(classes),
        "predicates": sorted(predicates),
        "properties": sorted(properties),
        "relationships": relationships,
    }

    OUT_SCHEMA.write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    print(f"Wrote schema to {OUT_SCHEMA}")


if __name__ == "__main__":
    build_schema()
