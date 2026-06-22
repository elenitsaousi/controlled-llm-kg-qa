from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.service import service


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ClarificationRequest(BaseModel):
    choiceId: str = Field(min_length=1)


app = FastAPI(title="True Demand KGQA API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("KGQA_ALLOWED_ORIGINS", "").split(",") if origin.strip()],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/api/questions")
def ask_question(payload: QuestionRequest):
    try:
        return service.ask(payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/autocomplete")
def autocomplete(q: str = Query(default=""), context: str = Query(default="")):
    return service.autocomplete(q, context)


@app.get("/api/examples")
def examples():
    return service.examples()


@app.get("/api/capabilities")
def capabilities():
    return list(service.answerable_capabilities())


@app.post("/api/clarifications/{case_id}")
def clarify(case_id: str, payload: ClarificationRequest):
    try:
        return service.clarify(case_id, payload.choiceId)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/graph/ontology")
def ontology():
    return service.ontology_payload()


@app.get("/api/graph/data")
def graph_data(limit: int = Query(default=500, ge=1, le=1000)):
    return service.data_payload(limit)


@app.get("/api/graph/evidence/{case_id}")
def evidence(case_id: str):
    return service.evidence_payload(case_id)


@app.get("/api/metrics/confidence")
def confidence_metrics():
    return service.metrics()


@app.get("/api/health")
def health():
    return service.health()
