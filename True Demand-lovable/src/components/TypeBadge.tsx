import type { EntityType } from "@/lib/types";

type Style = { bg: string; fg: string; border: string };

function mk(color: string, bgPct = 12, borderPct = 30): Style {
  return {
    bg: `color-mix(in oklab, ${color} ${bgPct}%, white)`,
    fg: color,
    border: `color-mix(in oklab, ${color} ${borderPct}%, white)`,
  };
}

const styles: Record<EntityType, Style> = {
  Class: mk("var(--type-class)"),
  ObjectProperty: mk("var(--type-object-property)"),
  DatatypeProperty: mk("var(--type-datatype-property)"),
  Property: mk("var(--type-property)"),
  Datatype: mk("var(--type-datatype)", 10, 28),
  Metric: mk("var(--type-metric)"),
  Dimension: mk("var(--type-dimension)"),
  Scope: {
    bg: "color-mix(in oklab, var(--type-scope) 18%, white)",
    fg: "color-mix(in oklab, var(--type-scope) 70%, black)",
    border: "color-mix(in oklab, var(--type-scope) 35%, white)",
  },
  Entity: mk("var(--type-entity)", 8, 25),
  Literal: { bg: "#fffbeb", fg: "#92400e", border: "#fde68a" },
};

const labels: Partial<Record<EntityType, string>> = {
  ObjectProperty: "Object property",
  DatatypeProperty: "Datatype property",
};

export function TypeBadge({ type }: { type: EntityType }) {
  const s = styles[type];
  const label = labels[type] ?? type;
  return (
    <span
      className="type-badge"
      style={{ backgroundColor: s.bg, color: s.fg, borderColor: s.border }}
    >
      {label}
    </span>
  );
}

const colors: Record<EntityType, string> = {
  Class: "#0e8a9c",
  ObjectProperty: "#64748b",
  DatatypeProperty: "#b7791f",
  Property: "#2f9e6e",
  Datatype: "#94a3b8",
  Metric: "#4a72c4",
  Dimension: "#8e5cc4",
  Scope: "#c79234",
  Entity: "#5b6b7a",
  Literal: "#d97706",
};

export function typeColor(type: EntityType): string {
  return colors[type] ?? "#5b6b7a";
}
