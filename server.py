

from __future__ import annotations

import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load a .env file (GROQ_API_KEY=..., optionally CHROMA_DIR=...) if one
# exists next to this file, so a real API key doesn't depend on getting
# shell syntax right (export vs set vs $env:) or on the variable surviving
# into whatever process launches uvicorn. Safe no-op if no .env is present
# -- a real environment variable set another way still takes priority.
load_dotenv()

from core import llm, pipeline
from core.parser import Transcript, parse_transcript
from core.store import Store
from interview_guide import DEFAULT_PROJECT_OBJECTIVE, DEFAULT_QUESTIONS

app = FastAPI(title="Expert Interview Analyzer API")

# Local dev frontend served from a different origin (e.g. a bundler) would
# need this; harmless when frontend is served from the same origin below.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB-backed persistence -- survives a server restart. See core/store.py
# for the schema. Data lives on disk under CHROMA_DIR (default ./chroma_data).
store = Store()


# --------------------------------------------------------------------------
# Serialization helpers -- turn dataclasses into plain JSON-able dicts
# --------------------------------------------------------------------------
def _serialize_expert_analysis(ea) -> dict:
    return {
        "expert_label": ea.expert_label,
        "display_name": ea.display_name,
        "results": [
            {
                "question": r.question,
                "status": r.status,
                "answer": r.answer,
                "citations": r.citations,
                "quotes": r.quotes,
            }
            for r in ea.results
        ],
    }


def _serialize_transcript(t: Transcript) -> dict:
    return {
        "expert_label": t.expert_label,
        "source_filename": t.source_filename,
        "has_timestamps": t.has_timestamps,
        "segments": [
            {
                "id": s.id,
                "index": s.index,
                "timestamp": s.timestamp,
                "speaker": s.speaker,
                "text": s.text,
                "label": s.citation_label(),
            }
            for s in t.segments
        ],
    }


# --------------------------------------------------------------------------
# API routes (registered before the static-file mount at "/", so they take
# precedence over it)
# --------------------------------------------------------------------------
@app.get("/api/default-guide")
def get_default_guide():
    return {"objective": DEFAULT_PROJECT_OBJECTIVE, "questions": DEFAULT_QUESTIONS}


@app.get("/api/models")
def get_models():
    return {"default": llm.DEFAULT_MODEL, "options": [llm.DEFAULT_MODEL] + llm.FALLBACK_MODELS}


@app.post("/api/analyze")
async def analyze(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    file3: UploadFile = File(...),
    name1: str = Form("Expert 1"),
    name2: str = Form("Expert 2"),
    name3: str = Form("Expert 3"),
    objective: str = Form(DEFAULT_PROJECT_OBJECTIVE),
    questions: str = Form("\n".join(DEFAULT_QUESTIONS)),
    model: str = Form(llm.DEFAULT_MODEL),
    api_key: Optional[str] = Form(None),
):
    try:
        client = llm.get_client(api_key or None)
    except llm.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))

    files = [file1, file2, file3]
    names = [name1, name2, name3]
    q_list = [q.strip() for q in questions.splitlines() if q.strip()]
    if not q_list:
        raise HTTPException(status_code=400, detail="No interview-guide questions provided.")

    transcripts: List[Transcript] = []
    for i, f in enumerate(files):
        raw_bytes = await f.read()
        raw = raw_bytes.decode("utf-8", errors="replace")
        t = parse_transcript(raw, expert_label=f"E{i+1}", filename=f.filename or f"expert{i+1}.txt")
        if not t.segments:
            raise HTTPException(status_code=400, detail=f"Could not extract any text from {names[i]}'s file.")
        transcripts.append(t)

    try:
        expert_analyses = []
        for i, t in enumerate(transcripts):
            ea = pipeline.answer_questions_for_expert(client, t, names[i], q_list, objective, model=model)
            expert_analyses.append(ea)

        cross = pipeline.cross_expert_analysis(client, expert_analyses, q_list, objective, model=model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Analysis failed: {e}")

    analysis_id = str(uuid.uuid4())
    store.save_analysis(
        analysis_id=analysis_id,
        project_objective=objective,
        questions=q_list,
        model=model,
        transcripts=transcripts,
        expert_analyses=expert_analyses,
        cross_analysis=cross,
    )

    return {
        "analysis_id": analysis_id,
        "experts": [_serialize_expert_analysis(ea) for ea in expert_analyses],
        "cross_analysis": cross,
        "transcripts": [_serialize_transcript(t) for t in transcripts],
    }


class ChatRequest(BaseModel):
    analysis_id: str
    question: str
    model: Optional[str] = None
    api_key: Optional[str] = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    data = store.load_analysis(req.analysis_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Unknown analysis_id -- run /api/analyze first.")

    try:
        client = llm.get_client(req.api_key or None)
    except llm.LLMError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = req.model or data.get("model") or llm.DEFAULT_MODEL

    try:
        result = pipeline.answer_chat_question(
            client,
            req.question,
            data["transcripts"],
            chat_history=data["chat_history"],
            model=model,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat failed: {e}")

    turn = {
        "question": req.question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "quotes": result.get("quotes", []),
        "answerable": result.get("answerable", True),
    }
    store.append_chat_turn(req.analysis_id, turn)
    return result


@app.get("/api/analyses")
def list_analyses():
    """Previously saved analyses (newest first) -- lets the frontend offer
    'resume' without re-uploading files or re-running any LLM calls."""
    return store.list_analyses()


@app.get("/api/analysis/{analysis_id}")
def get_analysis(analysis_id: str):
    data = store.load_analysis(analysis_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Unknown analysis_id.")
    return {
        "analysis_id": analysis_id,
        "created_at": data["created_at"],
        "project_objective": data["project_objective"],
        "questions": data["questions"],
        "model": data["model"],
        "experts": [_serialize_expert_analysis(ea) for ea in data["expert_analyses"]],
        "cross_analysis": data["cross_analysis"],
        "transcripts": [_serialize_transcript(t) for t in data["transcripts"]],
        "chat_history": data["chat_history"],
    }


@app.delete("/api/analysis/{analysis_id}")
def delete_analysis(analysis_id: str):
    store.delete_analysis(analysis_id)
    return {"deleted": True}


# --------------------------------------------------------------------------
# Static frontend (vanilla JS + Tailwind via CDN, no build step)
# Mounted last so the /api/* routes above take precedence.
# --------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
