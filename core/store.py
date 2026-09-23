"""
Persistence layer backed by ChromaDB.

Chroma's PersistentClient stores everything as local files under CHROMA_DIR
(default "./chroma_data"), so an analysis survives a server restart with no
separate database process to run. Four collections cover a full analysis:

  analyses         -- one row per analysis: guide, model, cross-analysis
  segments         -- one row per transcript segment (the citation source)
  expert_answers   -- one row per (expert, question) result
  chat_turns       -- one row per chat message, in order

Chroma is used here purely as a durable, filterable document/metadata store,
not yet for semantic search -- every add/upsert call supplies its own
placeholder embedding rather than pulling in a real embedding model, so
persistence stays dependency-light and works fully offline. Swapping in
real segment embeddings later (for the "retrieval instead of
context-stuffing" item in the README's scaling section) only touches this
file: the `segments` schema below is already exactly what that upgrade
would query against.

Chroma metadata values must be flat scalars (str/int/float/bool), so
anything structured (citations, quotes, question lists) is JSON-encoded on
the way in and decoded on the way out.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb

from .parser import Segment, Transcript
from .pipeline import ExpertAnalysis, QuestionResult

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_data")

_PLACEHOLDER_DIM = 8


def _placeholder_embedding() -> List[float]:
    # Real embeddings aren't needed for pure persistence/filtering; a fixed
    # dummy vector avoids downloading an embedding model just to store data.
    return [0.0] * _PLACEHOLDER_DIM


class Store:
    def __init__(self, path: str = CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=path)
        self.analyses = self.client.get_or_create_collection("analyses")
        self.segments = self.client.get_or_create_collection("segments")
        self.expert_answers = self.client.get_or_create_collection("expert_answers")
        self.chat_turns = self.client.get_or_create_collection("chat_turns")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    def save_analysis(
        self,
        analysis_id: str,
        project_objective: str,
        questions: List[str],
        model: str,
        transcripts: List[Transcript],
        expert_analyses: List[ExpertAnalysis],
        cross_analysis: list,
    ) -> None:
        names = [ea.display_name for ea in expert_analyses]
        labels = [t.expert_label for t in transcripts]

        self.analyses.upsert(
            ids=[analysis_id],
            documents=[project_objective + "\n\n" + "\n".join(questions)],
            embeddings=[_placeholder_embedding()],
            metadatas=[
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "project_objective": project_objective,
                    "questions": json.dumps(questions),
                    "model": model,
                    "expert_names": json.dumps(names),
                    "expert_labels": json.dumps(labels),
                    "source_filenames": json.dumps([t.source_filename for t in transcripts]),
                    "cross_analysis": json.dumps(cross_analysis),
                }
            ],
        )

        seg_ids, seg_docs, seg_metas = [], [], []
        for t, name in zip(transcripts, names):
            for s in t.segments:
                seg_ids.append(f"{analysis_id}::{s.id}")
                seg_docs.append(s.text)
                seg_metas.append(
                    {
                        "analysis_id": analysis_id,
                        "expert_label": t.expert_label,
                        "display_name": name,
                        "segment_id": s.id,
                        "index": s.index,
                        "timestamp": s.timestamp or "",
                        "speaker": s.speaker or "",
                        "source_filename": t.source_filename,
                    }
                )
        if seg_ids:
            self.segments.upsert(
                ids=seg_ids,
                documents=seg_docs,
                embeddings=[_placeholder_embedding() for _ in seg_ids],
                metadatas=seg_metas,
            )

        ans_ids, ans_docs, ans_metas = [], [], []
        for ea in expert_analyses:
            for i, r in enumerate(ea.results):
                ans_ids.append(f"{analysis_id}::{ea.expert_label}::q{i}")
                ans_docs.append(r.answer or "")
                ans_metas.append(
                    {
                        "analysis_id": analysis_id,
                        "expert_label": ea.expert_label,
                        "display_name": ea.display_name,
                        "order": i,
                        "question": r.question,
                        "status": r.status,
                        "citations": json.dumps(r.citations),
                        "quotes": json.dumps(r.quotes),
                    }
                )
        if ans_ids:
            self.expert_answers.upsert(
                ids=ans_ids,
                documents=ans_docs,
                embeddings=[_placeholder_embedding() for _ in ans_ids],
                metadatas=ans_metas,
            )

    def append_chat_turn(self, analysis_id: str, turn: dict) -> None:
        existing = self.chat_turns.get(where={"analysis_id": analysis_id})
        order = len(existing["ids"]) if existing and existing.get("ids") else 0
        self.chat_turns.upsert(
            ids=[f"{analysis_id}::chat::{order}"],
            documents=[turn.get("question", "")],
            embeddings=[_placeholder_embedding()],
            metadatas=[
                {
                    "analysis_id": analysis_id,
                    "order": order,
                    "answer": turn.get("answer", ""),
                    "citations": json.dumps(turn.get("citations", [])),
                    "quotes": json.dumps(turn.get("quotes", [])),
                    "answerable": bool(turn.get("answerable", True)),
                }
            ],
        )

    # ----------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------
    def load_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        got = self.analyses.get(ids=[analysis_id])
        if not got["ids"]:
            return None
        meta = got["metadatas"][0]

        questions = json.loads(meta["questions"])
        names = json.loads(meta["expert_names"])
        labels = json.loads(meta["expert_labels"])
        filenames = json.loads(meta.get("source_filenames", "[]") or "[]")
        cross_analysis = json.loads(meta["cross_analysis"])

        seg_rows = self.segments.get(where={"analysis_id": analysis_id})
        by_label: Dict[str, List[dict]] = {}
        for doc, sm in zip(seg_rows["documents"], seg_rows["metadatas"]):
            by_label.setdefault(sm["expert_label"], []).append({**sm, "text": doc})

        transcripts: List[Transcript] = []
        for label, filename in zip(labels, filenames):
            rows = sorted(by_label.get(label, []), key=lambda r: r["index"])
            segs = [
                Segment(
                    id=r["segment_id"],
                    index=r["index"],
                    timestamp=r["timestamp"] or None,
                    speaker=r["speaker"] or None,
                    text=r["text"],
                )
                for r in rows
            ]
            transcripts.append(
                Transcript(
                    expert_label=label,
                    segments=segs,
                    has_timestamps=any(s.timestamp for s in segs),
                    source_filename=filename,
                )
            )

        ans_rows = self.expert_answers.get(where={"analysis_id": analysis_id})
        by_expert: Dict[str, List[dict]] = {}
        for doc, am in zip(ans_rows["documents"], ans_rows["metadatas"]):
            by_expert.setdefault(am["expert_label"], []).append({**am, "answer": doc})

        expert_analyses: List[ExpertAnalysis] = []
        for label, name in zip(labels, names):
            rows = sorted(by_expert.get(label, []), key=lambda r: r["order"])
            results = [
                QuestionResult(
                    question=r["question"],
                    status=r["status"],
                    answer=r["answer"],
                    citations=json.loads(r["citations"]),
                    quotes=json.loads(r["quotes"]),
                )
                for r in rows
            ]
            expert_analyses.append(ExpertAnalysis(expert_label=label, display_name=name, results=results))

        chat_rows = self.chat_turns.get(where={"analysis_id": analysis_id})
        turns = []
        for doc, cm in zip(chat_rows["documents"], chat_rows["metadatas"]):
            turns.append(
                {
                    "_order": cm["order"],
                    "question": doc,
                    "answer": cm["answer"],
                    "citations": json.loads(cm["citations"]),
                    "quotes": json.loads(cm["quotes"]),
                    "answerable": cm.get("answerable", True),
                }
            )
        turns.sort(key=lambda t: t["_order"])
        for t in turns:
            t.pop("_order", None)

        return {
            "project_objective": meta["project_objective"],
            "questions": questions,
            "model": meta["model"],
            "created_at": meta["created_at"],
            "transcripts": transcripts,
            "expert_analyses": expert_analyses,
            "cross_analysis": cross_analysis,
            "chat_history": turns,
        }

    def list_analyses(self) -> List[dict]:
        got = self.analyses.get()
        rows = []
        for aid, meta in zip(got["ids"], got["metadatas"]):
            rows.append(
                {
                    "analysis_id": aid,
                    "created_at": meta["created_at"],
                    "project_objective": meta["project_objective"],
                    "expert_names": json.loads(meta["expert_names"]),
                    "model": meta["model"],
                }
            )
        rows.sort(key=lambda r: r["created_at"], reverse=True)
        return rows

    def delete_analysis(self, analysis_id: str) -> None:
        self.analyses.delete(ids=[analysis_id])
        for coll in (self.segments, self.expert_answers, self.chat_turns):
            existing = coll.get(where={"analysis_id": analysis_id})
            if existing and existing.get("ids"):
                coll.delete(ids=existing["ids"])
