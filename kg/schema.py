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

    def label_allowed(self, label: str) -> bool:
        return label in self.labels

    def relationship_allowed(
        self,
        rel_type: str,
        from_label: Optional[str] = None,
        to_label: Optional[str] = None,
    ) -> bool:
        for rel in self.relationships:
            if rel.get("type") != rel_type:
                continue
            if from_label and from_label not in rel.get("from", []):
                continue
            if to_label and to_label not in rel.get("to", []):
                continue
            return True
        return False

    def property_allowed(self, label: str, prop: str) -> bool:
        return prop in self.allowed_property_filters.get(label, [])

    def as_prompt_text(self) -> str:
        lines: List[str] = []
        if self.description:
            lines.append(f"Schema description: {self.description}")
        lines.append("Labels and properties:")
        for label, meta in sorted(self.labels.items()):
            props = meta.get("properties", [])
            props_text = ", ".join(props) if props else "none"
            lines.append(f"- {label}({props_text})")
        lines.append("Relationships (directed):")
        for rel in self.relationships:
            from_labels = ", ".join(rel.get("from", []))
            to_labels = ", ".join(rel.get("to", []))
            lines.append(f"- {from_labels} -[:{rel.get('type')}]→ {to_labels}")
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
    schema_path = os.path.join(base_dir, "data", "toy_kg", "schema.json")
    return load_schema(schema_path)
