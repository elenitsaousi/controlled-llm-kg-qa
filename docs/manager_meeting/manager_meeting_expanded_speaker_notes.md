# Expanded Speaker Notes: True Demand KGQA Manager Meeting

These notes are written to be copied into the PowerPoint speaker-notes field. They explain each slide in more detail than the visible slide text.

---

## Slide 1: Title

Today I will present the current state of my master thesis work on the True Demand Knowledge Graph Question Answering system. The purpose of the work is to make the True Demand knowledge graph usable by business or domain users who do not know RDF, SPARQL, or the internal ontology structure.

The system is designed as a hybrid KGQA pipeline. This means it does not simply send every question to an LLM and trust the first generated SPARQL query. Instead, it first checks whether the question can be answered through a deterministic graph-supported path. If the graph path is known, the system answers directly. If the question is ambiguous or unsupported by a deterministic route, it uses the LLM to generate candidate SPARQL queries, validates them, reranks them with guarded ML, and then decides whether to answer or ask for clarification.

For Infineon, the value is practical: the system can reduce manual graph exploration, make survey-grounded demand information easier to access, and provide an auditable interface where each answer can be traced back to graph evidence. It also reduces LLM usage by avoiding unnecessary LLM calls for questions that the graph can already answer deterministically.

The main research distinction I focus on is between candidate generation and candidate selection. Generation asks: did the LLM produce the correct query somewhere in the candidate set? Selection asks: did the system choose the correct query? In KGQA, generation can be strong while selection remains difficult, because several SPARQL queries can look valid but answer different business questions.

---

## Slide 2: Why This Work Matters

The business problem is that True Demand data is valuable but difficult to access directly. The data is represented in RDF and queried with SPARQL, which is powerful but not realistic for most non-technical users. A planner or business stakeholder should be able to ask questions such as "How does future demand change by region?" without knowing the exact class names, property names, or joins in the graph.

The technical problem is that a naive LLM-to-SPARQL pipeline is not reliable enough. An LLM may generate multiple queries that are syntactically valid and schema-valid, but they can still answer different questions. For example, "demand by region" could refer to current demand, future demand, OEM demand, semiconductor demand, or total regional demand. These are all related, but they are not the same interpretation.

This is why the hard problem is not only generation. The hard problem is selecting the query that matches the intended metric, dimension, aggregation, and scope. If the system selects a query that counts records when the user asked for a grouped demand breakdown, the answer may be graph-backed but semantically wrong.

The thesis therefore contributes both to Infineon's practical use case and to the research problem of reliable query selection under ambiguity. The result is a system that is more controlled, more auditable, and more cost-aware than an all-LLM interface.

---

## Slide 3: Data Sources Used

This slide summarizes the main artefacts used by the system.

The first artefact is the True Demand RDF graph. This is the analytical data layer. It contains more than 1.16 million triples and more than 100,000 entities. It stores survey-grounded and analytical information about demand, shortages, inventories, vehicle sales, regions, quarters, technology categories, vehicle types, companies, components, and autonomous-driving development.

The second artefact is the True Demand schema abstraction. This is not the full raw graph. It is a compact representation of the classes, properties, and supported relationships that the system uses for prompting, validation, routing, and ML features. This prevents the LLM from seeing an unstructured dump of the entire graph and helps the system reason over known graph paths.

The third artefact is the Digital Reference ontology. This is different from the True Demand data graph. The DR ontology provides conceptual and model-level information: definitions, labels, classes, object properties, datatype properties, domain and range information, and class hierarchy. I integrated it so the system can answer terminology questions such as "What is a Technology Node?" or "What does is processed by mean?" deterministically.

The fourth artefact is the benchmark data. I created and evaluated a final mixed benchmark of 1000 questions. This includes KG analytics questions, ontology definition questions, and advisory questions. I also built audit labels so the final accuracy is based on answer-level correctness, not only automatic execution success.

The important point is that the KG and the ontology have different roles. The KG contains data instances and analytical values. The ontology provides vocabulary and meaning. The final system connects both, but it does not confuse them.

---

## Slide 4: What the System Can Answer

The system covers three broad types of user questions.

The first type is KG analytics. These are numerical or analytical questions over the True Demand graph. Examples include future demand by region, current demand by survey group, shortage counts by survey type, vehicle sales by month, autonomous-driving percentages by vehicle type and SAE level, and inventory trends by component or technology category.

The second type is ontology definition questions. These are questions about what a concept or property means. For example, "What is Future Demand?", "Define Demand Class", or "What does has customer part number specification mean?" These questions should not be routed to analytics. They should be answered from the Digital Reference ontology or the deterministic glossary/ontology layer.

The third type is advisory questions. These are graph-grounded recommendation-style questions, such as "Which region should be monitored more closely?" or "What should I inspect first to understand shortage exposure?" These are not free-form business advice generated by the LLM. They are conservative templates based on query outputs and graph evidence.

The reason for separating these families is reliability. A definition question, an analytical aggregation question, and an advisory question require different execution logic. If the router confuses them, the system may return a valid-looking answer that is not what the user asked.

---

## Slide 5: User Interface Entry Points

The UI is designed for users with different levels of confidence and technical knowledge.

The free-text input allows users to type any natural-language question. This is useful for exploration, but it is also the riskiest path because the request may be ambiguous or unsupported. Therefore free text goes through request routing and confidence checks.

The validated examples are pre-tested questions. These should execute successfully against the current graph. They are useful for demos and for helping users understand what the system can answer.

The guided builder is more controlled. Instead of letting the user combine arbitrary concepts, it should offer only graph-supported combinations: metric, dimension, aggregation, and scope that correspond to a known path. This reduces invalid questions and improves trust.

The available topics section helps with discoverability. It shows the user which capabilities and dimensions exist, so they do not need to guess the ontology.

The Digital Reference browser is for ontology exploration. The user can search ontology terms, inspect definitions, and generate deterministic definition questions without using the LLM.

Finally, after an answer, the evidence graph helps the user see which graph entities and relationships are related to the answer. This is important for trust and for non-technical graph exploration.

---

## Slide 6: System Architecture

The system architecture has four main layers.

The first layer is the Streamlit UI. It collects the user's question, supports examples and the guided builder, provides the DR ontology browser, and displays the final answer and evidence graph.

The second layer is request routing. This determines whether the question is a KG analytics question, an ontology definition question, an advisory question, unsupported, or ambiguous. This is important because each type of request should be handled differently.

The third layer is deterministic routing. If the route is known and graph-supported, the system skips the LLM. For example, a DR ontology definition question goes directly to ontology lookup. A supported KG question goes to a direct SPARQL template. A supported advisory question goes to a conservative graph-backed advisory template.

The fourth layer is LLM fallback. If deterministic routing is insufficient, the system generates top-k SPARQL candidates using the LLM. It then validates candidates for syntax, schema compatibility, answer shape, scope, execution evidence, and safety flags. Guarded ML reranking is used to choose the best candidate, and confidence routing decides whether to answer automatically or ask for clarification.

The final layer is execution and output. SPARQL is executed against Fuseki or the RDF graph backend, the answer is synthesized based on the row shape, and audit logs record the route, query, rows, timing, and evidence.

The important architectural point is that this is not one monolithic LLM call. It is a controlled pipeline where the LLM is only one component.

---

## Slide 7: Deterministic Routing

Deterministic routing is used when the system can resolve the user's intent with high specificity. That means the capability is known, the metric is known, the dimension is supported, the aggregation is clear, the scope is clear enough, and the graph path is known.

For example, "Show future demand by region" can be deterministic if the capability is future demand, the dimension is region, and the intended output is a grouped breakdown. The system should not ask the user to choose between unrelated queries such as counting FutureDemandAnalysis entries or finding OEM share if those are not alternative interpretations of the question.

The deterministic route is implemented through a capability inventory and direct templates. It is not just keyword matching. It also checks whether the requested graph path is supported and whether the result has graph evidence.

The strict version of deterministic routing is important. Previously, deterministic coverage could be higher, but some direct answers were wrong because the system answered even when the path was not sufficiently safe. After tightening the guards, deterministic routing handles 514 out of 1000 questions and achieves 98.4% accuracy.

So the principle is: where deterministic works, it should work very well. If the system cannot guarantee the graph path, it should fall back rather than force an answer.

---

## Slide 8: When the System Falls Back to the LLM

The LLM fallback is used only when deterministic routing is not safe enough.

This can happen when there is no exact graph-supported template, when the question is genuinely ambiguous, when multiple interpretations remain possible, when the requested metric or dimension is not directly supported, or when a candidate query would return no rows or violate the requested answer shape.

Examples include complex autonomous-driving questions that combine vehicle type, SAE level, year, and survey provenance. Another example is current demand with BL1 and BL2 constraints for Tier1 Automotive. These questions require more than a direct template because the correct graph path is compositional.

This fallback route is where LLM generation and selection happen. The LLM generates candidate SPARQL queries. Then the system validates and reranks them. The LLM is useful because it can propose candidate query structures, but it is not trusted blindly.

The reason this is justified is that many fallback questions are difficult even for a human. A human would often need to inspect the ontology, check example triples, test SPARQL joins, and verify whether the result is semantically correct.

---

## Slide 9: Candidate Generation vs Selection

This slide explains an important evaluation distinction.

Candidate generation measures whether the correct query exists somewhere in the generated candidate set. Candidate selection measures whether the system chooses the correct candidate as the final query. Final answer correctness measures whether the user-facing answer is actually correct.

These are not the same. For example, the LLM may generate eight candidates and one of them may be correct. That means generation succeeded. But if the system selects a different candidate, selection failed. Conversely, sometimes a selected query may not exactly match the canonical query but still returns the same correct answer, so final answer correctness can be higher than strict selection accuracy.

This distinction is essential for the thesis. It shows that the bottleneck is often not that the LLM cannot generate a valid query. The bottleneck is deciding which graph-supported interpretation matches the user's intent.

This also explains why I report both selection-level metrics and final answer-level accuracy. Selection-level metrics tell us about the ranking problem. Answer-level accuracy tells us what the user experiences.

---

## Slide 10: Confidence and Ambiguity Routing

The system uses confidence and ambiguity signals to decide whether to answer, rerank, clarify, or avoid forced selection.

The signals include candidate score, score margin between the top candidates, normalized entropy over candidate scores, schema and answer-shape validation, execution evidence, and safety flags.

The policy is simple conceptually. If confidence is high and no safety flags are present, the system can answer. If ambiguity is moderate, ML and semantic reranking help choose among competing candidates. If ambiguity is high or the safety checks detect problems, the system should ask for clarification or avoid answering.

This is directly connected to the research idea: ambiguity is not uniform. Some questions have one dominant interpretation. Some have several plausible interpretations. The system should adapt its behavior instead of always forcing a top-1 query.

In the UI, this is reflected through clarification cards when needed. The goal is not to put the user in a loop unnecessarily. Clarification should only happen when there are genuine alternative interpretations, not merely because the graph supports other unrelated queries.

---

## Slide 11: Final Benchmark Design

The final benchmark contains 1000 user-facing questions.

There are 800 KG analytics questions. These test the main True Demand analytical functionality: demand, shortages, inventory, sales, autonomous driving, time periods, survey groups, regions, and vehicle types.

There are 150 Digital Reference ontology definition questions. These test whether the system can answer conceptual questions deterministically, such as definitions and property meanings.

There are 50 advisory questions. These test whether the system can produce conservative graph-grounded guidance from data, rather than only raw tables.

This mixed benchmark is important because a real user does not only ask one type of question. In practice, users will ask analytical questions, definition questions, and higher-level questions about what to inspect. The final evaluation therefore measures the complete user-facing system, not just an isolated LLM query generator.

---

## Slide 12: Evaluation Methodology

The evaluation was done end-to-end.

First, I ran all 1000 questions through the full system. For each question, the system logged the selected route, selected source, selected query, graph row count, row preview, answer text, execution reason, and timing.

Second, I built an audit CSV. This allowed me to inspect each answer together with the question and the evidence.

Third, I manually labeled the final answer as correct or incorrect. I checked whether the answer matched the requested metric, aggregation, grouping dimension, survey scope, and answer shape. For example, if the question asked for a grouped breakdown but the answer returned only the highest value, I labeled it incorrect.

Fourth, for incorrect answers, I classified how hard it would be for a schema-aware human to write the correct SPARQL query. Easy means the mismatch is obvious and the correct query is straightforward. Medium means the correct query requires schema awareness. Hard means the query requires graph exploration, indirect joins, or careful interpretation of how the KG is modeled.

This is why the final 90.8% is an audited answer-level accuracy, not an automatic proxy metric.

---

## Slide 13: Final Accuracy by Route

The final audited accuracy is 908 correct answers out of 1000, which is 90.8%.

The deterministic route handled 514 questions. Out of those, 506 were correct, giving 98.4% accuracy. This is the strongest part of the system because these are questions where the graph path is already known and supported.

The LLM fallback route handled 486 questions. Out of those, 402 were correct, giving 82.7% accuracy. This is lower, but expected, because this subset contains the harder questions that could not be answered safely with deterministic templates.

The important interpretation is that the system improves reliability by separating easy/supported cases from hard/ambiguous ones. It does not claim that the LLM alone reaches 90.8%. The full hybrid system reaches 90.8%.

---

## Slide 14: Accuracy by Question Family

The KG analytics subset contains 800 questions and reached 710 correct, or 88.8%. This is the hardest subset because it requires selecting the correct metric, dimension, aggregation, and scope over the graph.

The DR ontology definition subset contains 150 questions and reached 150 correct, or 100%. This is because these questions are routed deterministically to ontology lookup rather than the LLM.

The advisory subset contains 50 questions and reached 48 correct, or 96%. These answers are generated through conservative graph-backed templates. They are not arbitrary LLM advice.

This breakdown is important because the overall 90.8% is not a single homogeneous metric. Different question families have different difficulty and different execution paths.

---

## Slide 15: Correct vs Incorrect by Route

This slide shows where the wrong answers are concentrated.

The deterministic route produced only 8 incorrect answers out of 514. That means the strict deterministic routing policy is working well. It answers directly only when a trusted path is available.

The LLM fallback produced 84 incorrect answers out of 486. This is where most remaining errors occur. That is expected because this route receives the unresolved and more ambiguous questions.

This slide is useful for explaining why deterministic coverage and deterministic precision are both important. If deterministic routing is too broad, it may answer more questions but introduce false confidence. If it is too strict, it will push too many questions to the LLM. The current strict version prioritizes reliability.

---

## Slide 16: Cost and Latency

The evaluation run used cached LLM outputs, so the raw log reports zero fresh LLM calls. For the thesis and business interpretation, the fair number is the cold-run estimate.

In an all-LLM baseline, all 1000 questions would require an LLM call. At 20 cents per call, that would cost EUR 200.

In the hybrid system, 514 questions are answered without the LLM and 486 go to the fallback route. At 20 cents per fallback call, the estimated cold cost is EUR 97.20. This saves EUR 102.80 and reduces LLM calls by 51.4%.

Latency also improves for deterministic questions. The deterministic route averaged around 142 ms in the cached evaluation, while the cached LLM fallback averaged around 1819 ms. The direct route is therefore not only cheaper, but also faster for the user.

---

## Slide 17: Remaining Errors by Human Difficulty

There are 92 incorrect answers in the final audit.

I classified them by how hard it would be for a schema-aware human to write the correct SPARQL query. Only 8 are easy. These are cases where the system made a mistake that a human could probably avoid quickly.

There are 21 medium cases. These require schema awareness and careful interpretation, but they are not extremely difficult.

There are 63 hard cases. These are graph-composition problems where a human would likely need to inspect the ontology, check actual triples, test different joins, and verify non-empty results.

This classification helps explain the remaining error profile. The system is not mainly failing on simple questions. Most errors are in structurally difficult areas of the KG.

---

## Slide 18: Remaining Error Families

The remaining errors are concentrated in a few families.

The largest family is autonomous-driving complex grouping, with 24 errors. These questions often combine vehicle type, SAE level, year, and sometimes survey provenance. This creates complex graph paths and multiple plausible groupings.

The second major family is current-demand baseline or scope, with 20 errors. These involve BL1/BL2, market segment, Tier1/OEM/semiconductor scope, and current-demand interpretation.

Vehicle-sales metric or dimension errors account for 14 cases. These often involve distinguishing total units, monthly/yearly aggregation, actual versus forecast, and vehicle-type dimensions.

Future-demand complex-dimension errors account for 12 cases. These involve combinations of future demand with quarter, region, technology category, and scope.

The key point is that future improvement should focus on these recurring graph families, not just generic prompt tuning.

---

## Slide 19: Examples of Correct Deterministic Answers

This slide should show concrete examples where deterministic routing works well.

For "What is a Technology Node?", the system routes to the Digital Reference ontology and returns a definition. No LLM call is needed.

For "Show future demand by region", the system resolves the Future Demand capability, the Region dimension, and the grouped breakdown intent. It executes the direct graph-supported SPARQL path.

For "How many OEM companies reported shortages?", the system uses the shortage template with OEM scope and returns a count from the graph.

For "Show monthly vehicle sales actual vs forecast", the system uses a vehicle-sales template grouped by month and sales type.

These examples show the practical benefit: common graph-supported questions can be answered quickly, cheaply, and consistently.

---

## Slide 20: Examples of Hard LLM Fallback Cases

This slide should show examples where the fallback route is justified.

One example is autonomous driving by vehicle type, SAE level, year, and survey provenance. The user-facing question may sound simple, but the graph path is complex because the dimensions are not always connected through a single straightforward relation.

Another example is current demand with BL1/BL2 and Tier1 Automotive scope. The query must preserve baseline, market segment, and survey origin. A query can easily be non-empty but still semantically wrong if one of these constraints is missing.

Future demand by quarter and technology category is also difficult because it combines time, technology category, and future-demand interpretation. The graph may support related paths, but not all combinations are equally valid.

The point is that LLM fallback errors are often understandable. They are not necessarily basic language failures. Many require graph modeling knowledge and iterative SPARQL testing.

---

## Slide 21: Digital Reference Integration

The Digital Reference integration adds ontology-model understanding to the system.

Before this, the system was mainly focused on True Demand analytics. Now, the user can also ask conceptual questions. This is important because non-technical users often need to understand what a term means before they can ask the right analytical question.

The DR route is deterministic. It searches ontology labels, local names, definitions, properties, domain/range, and hierarchy information. It does not call the LLM for definition questions.

In the final benchmark, the DR ontology definition subset had 150 questions and all 150 were correct. This shows that deterministic ontology lookup is reliable for this task.

The distinction is important: the DR ontology does not replace the True Demand KG. It complements it. The KG answers data questions; the ontology explains the model and terminology.

---

## Slide 22: Evidence and Auditability

Every answer is logged with route, selected source, selected query, graph rows, row preview, answer text, execution reason, and timing.

This matters because enterprise KGQA should be auditable. If a user or evaluator asks why the system answered something, we can inspect the actual query and the graph evidence.

The audit log also supports evaluation. It allowed me to manually label the 1000 benchmark answers and identify recurring failure patterns.

This is different from a black-box chatbot. The system keeps the selected query and evidence visible, making it possible to debug, verify, and improve the pipeline.

---

## Slide 23: What Improved Recently

The most important recent improvement was tightening deterministic routing.

Earlier, the system sometimes answered deterministically even when the graph path was not safe enough. That increased coverage but created false confidence. I changed the policy so deterministic routing fires only when the supported path, scope, and answer shape are clear.

I also improved the separation between ontology definitions and KG analytics. For example, "Define Future Demand" should not become a graph aggregation query. It should be routed to deterministic ontology lookup.

I improved several direct templates, including shortage scope handling, actual-versus-forecast vehicle sales, rank/top questions, and supported grouped breakdowns.

The result is that deterministic accuracy increased to 98.4%, while the system still falls back to the LLM when deterministic routing would be unsafe.

---

## Slide 24: Next Steps

The next improvements should be targeted.

First, I would focus on the recurring hard error families: autonomous-driving complex grouping, current-demand BL/scope, vehicle-sales dimensions, and future-demand multi-dimensional paths. These are the areas where most remaining errors occur.

Second, I would add deterministic templates only where the graph truly supports the path. I would avoid adding templates just to cover benchmark questions, because that would make the system biased and less general.

Third, I would improve answer synthesis for grouped and comparison results. Some errors are not only query errors; they are answer-shape or summarization issues.

Fourth, I would continue improving the evidence graph UI, so users can inspect the nodes and relationships behind an answer more easily.

Finally, I would keep the manual audit process for any final evaluation claim, because automatic correctness proxies are not sufficient for KGQA.

---

## Slide 25: Closing Message

The main conclusion is that a hybrid KGQA architecture is more reliable than a pure LLM-to-SPARQL approach for this enterprise knowledge graph.

The system uses deterministic routing where graph paths are known, which gives high accuracy, low latency, and lower cost. It uses the LLM only for unresolved or ambiguous questions, and even then it validates and reranks candidate queries instead of trusting the first output.

The final audited result is 90.8% answer-level accuracy on 1000 mixed questions. The deterministic route achieved 98.4%, the LLM fallback achieved 82.7%, DR ontology definitions achieved 100%, and the system reduced estimated LLM calls by 51.4% compared with an all-LLM baseline.

For Infineon, this shows that natural-language access to complex enterprise graph data is feasible, but only if the system is controlled, graph-grounded, and auditable.

