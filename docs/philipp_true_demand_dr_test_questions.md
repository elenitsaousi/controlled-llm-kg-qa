# True Demand KGQA Test Questions

This file lists representative questions that can be used to test the True Demand KGQA app. The system is not limited to these exact strings, but these questions cover the main supported areas: True Demand analytics, Digital Reference ontology lookup, graph/source metadata, advisory questions, and clarification behavior.

## 1. Source and Scope Questions

Expected behavior: deterministic answer, no LLM needed.

- What sources are loaded?
- What is the scope of my sources?
- Give me a brief summary of the sources.
- What does True Demand cover?
- Summarize the True Demand knowledge graph.
- What does the Digital Reference ontology cover?
- Summarize the Digital Reference source.
- What is the Digital Reference used for?
- How is the Digital Reference different from the True Demand graph?

## 2. Graph and Schema Metadata

Expected behavior: deterministic answer, no LLM needed.

- How many triples are in the graph?
- How many nodes do we have?
- How many entities are in the graph?
- How many classes are available?
- How many predicates are available?
- How many properties are available?
- What classes are available?
- What predicates are available?
- What properties are available?
- What can I ask?
- Which topics are covered?
- What question types are supported?
- Do you support demand by quarter?
- Do you support vehicle sales by month?
- Do you have shortage questions?
- Do you have ontology definition questions?

## 3. True Demand: Regional and Current Demand

Expected behavior: graph-backed answer. Many of these should use deterministic routes.

- Show current demand by region.
- List OEM total demand by region.
- Show total current demand from OEM customers by region.
- Show total current demand from Tier1 customers by region.
- Show total current demand from Semiconductor customers by region.
- Break down total regional demand by survey origin and region.
- Show regional demand by survey group.
- Can you show me the average quarterly demand percentage trend based on the OEM survey results?
- Can you show me the average demand percentage trend for the Tier1 survey by quarter?
- Can you show me the average quarterly percentage trend in demand for the Semiconductor survey?
- Show demand for the last 3 months.
- Show current demand for the last three months.
- How has current demand developed in the past three months?

## 4. True Demand: Future Demand

Expected behavior: graph-backed answer or relevant clarification if the question is too broad.

- Show future demand by region.
- Show the total percentage of future demand for OEM, detailed by quarter and region.
- Show the overall future demand for Tier1, grouped by region and quarter.
- Show the total percentage of future demand for Semiconductor, detailed by quarter and region.
- Show future semiconductor demand by technology category and quarter.
- Can you provide the total future demand for semiconductors segmented by technology category and quarter?
- What is the average percentage change in future demand broken down by vehicle type and quarter?
- What is the combined future demand for Option1, Option2, and Option3 in Automotive, broken down by quarter?
- Which vehicle type has the highest average future-demand change?
- Which technology categories account for the highest future semiconductor demand in each quarter?

## 5. True Demand: Current Demand Baselines

Expected behavior: graph-backed answer.

- Which percentage changes apply to Tier1 automotive for baselines BL1 and BL2?
- What is the average current-demand change for BL1 and BL2 products in the Tier1 Automotive segment?
- What is the total Tier1 current demand percentage change difference between BL1 and BL2?
- Compare BL1 and BL2 current-demand changes for Tier1 Automotive.

## 6. True Demand: Vehicle Sales

Expected behavior: graph-backed answer.

- Show vehicle sales by month.
- Show actual vehicle sales by month.
- What are the monthly vehicle sales totals from actual transactions?
- How do the forecasted vehicle unit totals break down by month?
- Compare actual and forecast vehicle-sales totals by month.
- Show me the difference between actual and forecasted vehicle sales totals broken down by month.
- Can you show the total number of vehicles sold each year, grouped by type?
- Which month has the highest actual vehicle sales?

## 7. True Demand: Inventory

Expected behavior: graph-backed answer.

- Summarize Tier1 inventory participant totals by component.
- What is the overall Tier1 inventory amount for each component and trend?
- Show inventory trends by component.
- For each semiconductor technology category and inventory trend, how many inventory entries are recorded?
- Show inventory by technology category.

## 8. True Demand: Shortages

Expected behavior: graph-backed answer or graph-grounded advisory answer for monitoring/focus questions.

- How many companies report shortage by survey type?
- How many companies have indicated shortages, grouped by the type of survey?
- What is the number of OEM companies with and without a shortage?
- How many Tier1 companies are experiencing a shortage compared to those that are not?
- How many semiconductor companies report a shortage versus no shortage?
- Review shortage exposure by survey group.
- Which survey group appears most exposed to shortage?

## 9. True Demand: Order Cancellation

Expected behavior: graph-backed answer.

- What is the total count of order cancellation responses per semiconductor technology category?
- Can you provide the total count of semiconductor order-cancellation responses grouped by technology category and response type?
- Summarize increase, decrease, and stable order-cancellation response trends by semiconductor technology category.
- Group order-cancellation participant counts by technology category and response type.

## 10. True Demand: Autonomous Driving

Expected behavior: graph-backed answer.

- What is the average autonomous driving development broken down by vehicle type and SAE level?
- What is the average autonomous driving development for OEMs by vehicle type, SAE level, and year?
- What is the average autonomous-driving development for Tier1 suppliers, grouped by vehicle type, SAE level, and year?
- For each year, what is the average autonomous driving development percentage?
- Which vehicle type makes up the largest percentage at Level 5 autonomy?

## 11. Catalog Lookup

Expected behavior: graph-backed lookup answer.

- What are the names of all regions recorded in our database?
- What are the names of all technology categories?
- What are the quarter labels present in our dataset?
- Can you tell me how many companies are currently listed?
- List companies that reported semiconductor shortage.

## 12. Digital Reference Ontology Questions

Expected behavior: deterministic ontology answer from the Digital Reference, no LLM needed when the term is found.

- What is a Technology Node?
- Define Future Demand.
- What is True Demand?
- What is a Semiconductor?
- What is a Product?
- What is a Customer?
- What is a Supply Chain?
- What is a Knowledge Graph?
- What is an Ontology?
- What is a lobe?
- What are single lobe and cross lobe?
- What does is processed by mean?
- What does acts on property mean?
- Explain the has part relationship.
- Define Technology Category.
- Define Demand.
- Define Component.
- Define Product, Customer, and Supply Chain.
- Define Technology Node and Current Demand.
- Define Single Lobe, Cross Lobe, and Technology Node.
- Define a class and an object property, for example Product and is processed by.
- Demand vs current demand.
- What is the difference between current demand and future demand?
- Compare single lobe and cross lobe.

## 13. Advisory Questions

Expected behavior: graph-grounded advisory route or relevant clarification asking which evidence view should be used.

- Which region should be monitored more closely based on current demand?
- Where should planning attention focus based on the survey data?
- Which demand area seems most uncertain?
- What should I look at if I want to understand future demand risk?
- Which survey group appears most exposed to shortage?
- Which technology category should be reviewed first based on future demand?

## 14. Ambiguous Questions for Clarification Testing

Expected behavior: relevant clarification options, not unrelated choices.

- Show me the demand.
- How is demand developing?
- What is the latest demand?
- Show semiconductor demand.
- Show demand by quarter.
- Show demand by region.
- Which area should I monitor?
- What should planning focus on?

## 15. Unsupported or Out-of-Scope Questions

Expected behavior: controlled no-answer or clarification, not hallucinated answers.

- What is the weather in Munich?
- What is the stock price of Infineon?
- Predict semiconductor demand for next year.
- What will happen in the market in 2030?
- Give me a business decision for China.
- Show demand for a product that is not in the graph.

## Notes for Testing

- The app is a domain-bounded KGQA system, not a general chatbot.
- The True Demand graph is the analytical source for numerical and graph-backed questions.
- The Digital Reference ontology is the definition and terminology source.
- If a question is broad but answerable through several graph paths, the system should ask for clarification.
- If a question cannot be answered from the graph or ontology, the system should say so instead of inventing an answer.
- The current interactive target is that the app returns either an answer, a clarification, or a controlled no-answer within roughly 10 seconds.
