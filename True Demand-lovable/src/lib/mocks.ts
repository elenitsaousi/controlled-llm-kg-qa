import type { ConfidenceMetrics, GraphPayload, HealthStatus, QuestionResponse } from "./types";

export const mockSuggestions = [
  {
    id: "future-demand",
    label: "Future Demand",
    type: "Metric" as const,
    description: "Future-demand survey analysis.",
  },
  {
    id: "regional-demand",
    label: "Regional Demand",
    type: "Metric" as const,
    description: "Current demand by region.",
  },
  {
    id: "region",
    label: "Region",
    type: "Dimension" as const,
    description: "Graph-supported regional breakdown.",
  },
  {
    id: "quarter",
    label: "Quarter",
    type: "Dimension" as const,
    description: "Graph-supported quarterly breakdown.",
  },
];

export const mockExamples = [
  { id: "future-demand-region", text: "Show future demand by region.", category: "Future Demand" },
  { id: "vehicle-sales-month", text: "Show vehicle sales by month.", category: "Vehicle Sales" },
];

export const mockCapabilities = [
  {
    family: "Future Demand",
    templates: 2,
    description: "Graph-supported future-demand analysis.",
    dimensions: ["region", "quarter"],
    aggregations: ["AVG", "SUM"],
    examples: [mockExamples[0]],
  },
  {
    family: "Vehicle Sales",
    templates: 1,
    description: "Graph-supported vehicle-sales analysis.",
    dimensions: ["month"],
    aggregations: ["SUM"],
    examples: [mockExamples[1]],
  },
];

export const mockOntology: GraphPayload = {
  nodes: [
    { id: "FutureDemandAnalysis", label: "Future Demand Analysis", type: "Class" },
    { id: "Region", label: "Region", type: "Class" },
  ],
  edges: [
    { id: "for-region", source: "FutureDemandAnalysis", target: "Region", label: "in region" },
  ],
};

const evidenceByCase = new Map<string, GraphPayload>();

export function pickMockAnswer(question: string): QuestionResponse {
  const caseId = `mock-${Date.now()}`;
  evidenceByCase.set(caseId, mockOntology);
  return {
    caseId,
    decision: "direct",
    answer: `Mock graph answer for: ${question}`,
    confidence: 1,
    entropy: 0,
    responseTimeMs: 20,
    sparql: "SELECT ?region WHERE { ?entry survey:inRegion ?region . }",
    diagnostics: { family: "Future Demand", template: "mock", safetyFlags: [] },
  };
}

export function resolveClarification(_choiceId: string): QuestionResponse {
  return pickMockAnswer("clarified True Demand question");
}

export function mockEvidenceFor(caseId: string): GraphPayload {
  return evidenceByCase.get(caseId) ?? { nodes: [], edges: [] };
}

export const mockMetrics: ConfidenceMetrics = {
  forcedTop1: 0.677,
  anyCorrect: 0.944,
  autoAnswerAccuracy: 0.955,
  coverage: 0.674,
  systemAccuracy: 0.798,
  directAccuracy: 0.955,
  llmAccuracy: 0.472,
  confidenceBuckets: [],
  entropyRouting: [
    { route: "Direct Answer", count: 337 },
    { route: "LLM + Ranking", count: 163 },
  ],
  accuracyVsCoverage: [],
  cases: [],
};

export const mockHealth: HealthStatus = {
  api: "ok",
  fuseki: "degraded",
  llm: "degraded",
  version: "mock",
  latencyMs: 0,
};

export const mockEvidence: GraphPayload = { nodes: [], edges: [] };
export const mockAnswer: QuestionResponse = pickMockAnswer("Show future demand by region.");
