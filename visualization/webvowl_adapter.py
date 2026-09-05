from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from rdflib import Graph, Literal, RDF, RDFS, OWL, URIRef

SURVEY_NS = "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"
WEBVOWL_CACHE_DIR = Path(".cache") / "webvowl"


@dataclass
class VOWLConversionResult:
    ok: bool
    ontology_path: Path
    json_path: Optional[Path] = None
    message: str = ""
    stdout: str = ""
    stderr: str = ""


def _short_term(term) -> str:
    value = str(term)
    if value.startswith(SURVEY_NS):
        return value[len(SURVEY_NS) :]
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    if "/" in value:
        return value.rstrip("/").rsplit("/", 1)[-1]
    return value


def _safe_local_uri(value: object) -> URIRef:
    if isinstance(value, URIRef):
        return value
    text = str(value).strip()
    if text.startswith(("http://", "https://")):
        return URIRef(text)
    slug = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in text)
    slug = slug.strip("_") or "Node"
    return URIRef(SURVEY_NS + slug)


def write_relationship_slice_ontology(
    triples: Sequence[Tuple[object, object, object]],
    output_path: Path,
    *,
    ontology_iri: str = "http://example.org/true-demand/evidence-slice",
) -> Path:
    """Write a small OWL ontology slice from class-to-class relationship triples.

    WebVOWL is an ontology viewer. For answer evidence we therefore serialize
    relationships as OWL object properties with explicit domain/range classes,
    instead of sending raw observation triples.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = Graph()
    graph.bind("survey", SURVEY_NS)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)
    ontology = URIRef(ontology_iri)
    graph.add((ontology, RDF.type, OWL.Ontology))

    for source, predicate, target in triples:
        source_uri = _safe_local_uri(source)
        predicate_uri = _safe_local_uri(predicate)
        target_uri = _safe_local_uri(target)
        graph.add((source_uri, RDF.type, OWL.Class))
        graph.add((target_uri, RDF.type, OWL.Class))
        graph.add((predicate_uri, RDF.type, OWL.ObjectProperty))
        graph.add((predicate_uri, RDFS.domain, source_uri))
        graph.add((predicate_uri, RDFS.range, target_uri))
        graph.add((source_uri, RDFS.label, Literal(graph.namespace_manager.normalizeUri(source_uri).split(":")[-1])))
        graph.add((target_uri, RDFS.label, Literal(graph.namespace_manager.normalizeUri(target_uri).split(":")[-1])))
        graph.add((predicate_uri, RDFS.label, Literal(_short_term(predicate_uri))))

    graph.serialize(destination=str(output_path), format="turtle")
    return output_path


def _output_name_for_ontology(ontology_path: Path) -> str:
    digest = hashlib.sha1(ontology_path.read_bytes()).hexdigest()[:12]
    return f"{ontology_path.stem}_{digest}.json"


def convert_ontology_with_owl2vowl(
    *,
    ontology_path: Path,
    owl2vowl_jar_path: str,
    java_cmd: str = "java",
    cache_dir: Path = WEBVOWL_CACHE_DIR,
    timeout_s: int = 60,
) -> VOWLConversionResult:
    ontology_path = Path(ontology_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not ontology_path.exists():
        return VOWLConversionResult(
            ok=False,
            ontology_path=ontology_path,
            message=f"Ontology file not found: {ontology_path}",
        )
    jar = Path(str(owl2vowl_jar_path or "").strip())
    if not jar.exists():
        return VOWLConversionResult(
            ok=False,
            ontology_path=ontology_path,
            message=(
                "OWL2VOWL converter jar is not configured. Set OWL2VOWL_JAR_PATH "
                "or fill the path in Developer settings."
            ),
        )

    target = cache_dir / _output_name_for_ontology(ontology_path)
    if target.exists() and target.stat().st_mtime >= ontology_path.stat().st_mtime:
        return VOWLConversionResult(
            ok=True,
            ontology_path=ontology_path,
            json_path=target,
            message="Using cached VOWL JSON.",
        )

    command = [java_cmd, "-jar", str(jar), "-file", str(ontology_path), "-echo"]
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return VOWLConversionResult(
            ok=False,
            ontology_path=ontology_path,
            message=f"OWL2VOWL conversion failed to start: {exc}",
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        return VOWLConversionResult(
            ok=False,
            ontology_path=ontology_path,
            message=f"OWL2VOWL exited with status {proc.returncode}.",
            stdout=stdout,
            stderr=stderr,
        )

    json_payload = _extract_json(stdout)
    if not json_payload:
        return VOWLConversionResult(
            ok=False,
            ontology_path=ontology_path,
            message="OWL2VOWL did not return JSON on stdout.",
            stdout=stdout,
            stderr=stderr,
        )
    target.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return VOWLConversionResult(
        ok=True,
        ontology_path=ontology_path,
        json_path=target,
        message="Converted ontology to VOWL JSON.",
        stdout=stdout,
        stderr=stderr,
    )


def _extract_json(stdout: str) -> Optional[Dict[str, object]]:
    text = (stdout or "").strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def build_webvowl_iframe_html(webvowl_url: str, *, height_px: int = 760) -> str:
    url = (webvowl_url or "").strip() or "http://localhost:8080"
    return f"""
    <div style="border:1px solid #d3dde4;border-radius:8px;overflow:hidden;background:#f6f8fa;">
      <iframe
        src="{url}"
        width="100%"
        height="{max(420, int(height_px))}"
        style="border:0;display:block;background:#eef3f6;"
        title="WebVOWL ontology visualization"
      ></iframe>
    </div>
    """
