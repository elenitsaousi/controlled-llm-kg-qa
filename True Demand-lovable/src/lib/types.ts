export type EntityType =
  | "Class"
  | "ObjectProperty"
  | "DatatypeProperty"
  | "Property"
  | "Datatype"
  | "Metric"
  | "Dimension"
  | "Scope"
  | "Entity"
  | "Literal";

export type DecisionKind = "direct" | "auto" | "clarification" | "unsupported";

export interface Suggestion {
  id: string;
  label: string;
  type: EntityType;
  description?: string;
}

export interface Example {
  id: string;
  text: string;
  category?: string;
}

export interface Capability {
  family: string;
  templates: number;
  description: string;
  dimensions: string[];
  aggregations: string[];
  examples?: Example[];
}

export interface TableColumn {
  key: string;
  label: string;
}

export interface AnswerTable {
  columns: TableColumn[];
  rows: Record<string, string | number>[];
}

export interface Interpretation {
  id: string;
  text: string;
  confidence: number;
  /** Optional short technical hint, e.g. the SPARQL fragment differentiating this reading */
  hint?: string;
}

export interface QuestionResponse {
  caseId: string;
  decision: DecisionKind;
  answer: string;
  table?: AnswerTable;
  interpretations?: Interpretation[];
  confidence: number;
  entropy: number;
  responseTimeMs: number;
  sparql: string;
  /** When true, backend produced no evidence/rows; UI must show unsupported state */
  unsupported?: boolean;
  diagnostics: {
    family: string;
    template?: string;
    rankerScore?: number;
    safetyFlags?: string[];
  };
}

export interface GraphNode {
  id: string;
  label: string;
  type: EntityType;
  /** Full IRI, e.g. https://schema.truedemand.io/AURIX_TC4x */
  iri?: string;
  /** rdfs:comment / skos:definition */
  definition?: string;
  /** For Literal nodes */
  value?: string | number;
  datatype?: string;
  description?: string;
  properties?: { key: string; value: string }[];
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  /** Predicate IRI, e.g. https://schema.truedemand.io/trueDemand */
  iri?: string;
}

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Subset of node ids forming the SPARQL answer path */
  pathNodeIds?: string[];
  /** Subset of edge ids forming the SPARQL answer path */
  pathEdgeIds?: string[];
}

export interface ConfidenceMetrics {
  forcedTop1: number;
  anyCorrect: number;
  autoAnswerAccuracy: number;
  coverage: number;
  systemAccuracy?: number;
  directAccuracy?: number;
  llmAccuracy?: number;
  confidenceBuckets: { bucket: string; count: number; correct: number }[];
  entropyRouting: { route: string; count: number }[];
  accuracyVsCoverage: { coverage: number; accuracy: number }[];
  cases: CaseSummary[];
}

export interface CaseSummary {
  id: string;
  question: string;
  family: string;
  decision: DecisionKind;
  confidence: number;
  entropy: number;
  correct: boolean | null;
  safetyFlags: string[];
  ambiguity: number;
}

export interface HealthStatus {
  api: "ok" | "degraded" | "down";
  fuseki: "ok" | "degraded" | "down";
  llm: "ok" | "degraded" | "down";
  version: string;
  latencyMs: number;
}
