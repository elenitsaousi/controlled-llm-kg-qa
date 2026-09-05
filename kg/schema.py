import json
import os
from typing import Dict, List, Optional


class KGSchema:
    def __init__(self, schema_dict: Dict):
        self.description = schema_dict.get("description", "")
        self.labels = schema_dict.get("labels", {})
        self.relationships = schema_dict.get("relationships", [])
        self.allowed_property_filters = schema_dict.get(
            "allowed_property_filters", {}
        )
        self.notes = schema_dict.get("notes", [])

        # Ontology-centric fields (RDF/OWL)
        self.classes = list(schema_dict.get("classes", []))
        self.predicates = list(schema_dict.get("predicates", []))
        self.properties = list(schema_dict.get("properties", []))

        # Backward-compatible inference from legacy schema
        if not self.classes and self.labels:
            self.classes = list(self.labels.keys())
        if not self.predicates and self.relationships:
            self.predicates = [rel.get("type") for rel in self.relationships]
        if not self.properties and self.allowed_property_filters:
            props = set()
            for values in self.allowed_property_filters.values():
                props.update(values)
            self.properties = sorted(props)

        self._class_set = set(self.classes)
        self._predicate_set = set(self.predicates)
        self._property_set = set(self.properties)

    def class_allowed(self, cls: str) -> bool:
        return cls in self._class_set

    def predicate_allowed(self, pred: str) -> bool:
        return pred in self._predicate_set or pred in self._property_set

    def property_allowed(self, label: str, prop: str) -> bool:
        if label in self.allowed_property_filters:
            return prop in self.allowed_property_filters.get(label, [])
        return prop in self._property_set

    # Backwards-compatible aliases
    def label_allowed(self, label: str) -> bool:
        return self.class_allowed(label)

    def relationship_allowed(
        self,
        rel_type: str,
        from_label: Optional[str] = None,
        to_label: Optional[str] = None,
    ) -> bool:
        if self.relationships:
            for rel in self.relationships:
                if rel.get("type") != rel_type:
                    continue
                if from_label and from_label not in rel.get("from", []):
                    continue
                if to_label and to_label not in rel.get("to", []):
                    continue
                return True
            return False
        return self.predicate_allowed(rel_type)

    def as_prompt_text(self) -> str:
        lines: List[str] = []
        if self.description:
            lines.append(f"Ontology description: {self.description}")
        if self.classes:
            lines.append("Classes:")
            for cls in sorted(self.classes):
                lines.append(f"- {cls}")
        if self.predicates:
            lines.append("Predicates:")
            for pred in sorted(self.predicates):
                lines.append(f"- {pred}")
        if self.relationships:
            lines.append("Predicate Domains/Ranges:")
            for rel in self.relationships:
                rel_type = rel.get("type")
                from_labels = ", ".join(rel.get("from", []))
                to_labels = ", ".join(rel.get("to", []))
                if rel_type and (from_labels or to_labels):
                    lines.append(f"- {rel_type}: {from_labels} -> {to_labels}")
        if self.properties:
            lines.append("Properties:")
            for prop in sorted(self.properties):
                lines.append(f"- {prop}")
        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"- {note}")
        return "\n".join(lines)


def load_schema(schema_path: str) -> KGSchema:
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return KGSchema(data)


def load_default_schema() -> KGSchema:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    schema_path = os.path.join(base_dir, "data", "infineon", "schema.json")
    return load_schema(schema_path)
