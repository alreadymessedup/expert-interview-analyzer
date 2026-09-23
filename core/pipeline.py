"""
Orchestration layer: turns parsed Transcripts into structured, cited
analysis. Three entry points, matching the three "modes" of the app:

  1. answer_questions_for_expert() -- interview-guide Q&A for one transcript
  2. cross_expert_analysis()       -- themes & disagreements across experts
  3. answer_chat_question()        -- free-form Q&A across all transcripts

All three share the same grounding discipline:
  - the model only sees the transcript(s), never asked to use outside
    knowledge
  - every factual claim must carry a segment_id citation
  - "not discussed" is an explicit, expected, valid answer
  - quotes are re-verified in code against the source text (see verify.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from groq import Groq

from . import llm, verify
from .parser import Transcript

GROUNDING_RULES = """You are analyzing a real expert-interview transcript for a market research project.

Hard rules, no exceptions:
1. Use ONLY information explicitly present in the transcript provided below. Never use outside knowledge, never infer facts that are not stated, never fill gaps with plausible-sounding assumptions.
2. Every answer must cite the exact segment_id(s) it is based on, taken from the segment IDs shown in the transcript (e.g. "E1-S014"). Never invent a segment_id.
3. If the transcript does not address a question, or only touches it tangentially, say so explicitly. Do not stretch a tangential comment into a full answer.
4. Any "quote" you provide must be copied character-for-character from the transcript text of the cited segment. Do not paraphrase inside a quote. Do not combine words from two segments into one quote. Keep quotes short (under ~35 words) and pick the most substantive line, not filler.
5. If you are not confident a claim is supported by the transcript, leave it out rather than guessing.
"""


def _transcript_block(t: Transcript) -> str:
    lines = []
    for s in t.segments:
        speaker = f" {s.speaker}" if s.speaker else ""
        ts = f" {s.citation_label()}"
        lines.append(f"[{s.id}]{ts}{speaker}: {s.text}")
    return "\n".join(lines)


@dataclass
class QuestionResult:
    question: str
    status: str  # "answered" | "partial" | "not_discussed"
    answer: str
    citations: List[dict] = field(default_factory=list)  # [{"segment_id","timestamp"}]
    quotes: List[dict] = field(default_factory=list)  # [{"text","segment_id","timestamp","verified"}]


@dataclass
class ExpertAnalysis:
    expert_label: str
    display_name: str
    results: List[QuestionResult] = field(default_factory=list)


def answer_questions_for_expert(
    client: Groq,
    transcript: Transcript,
    display_name: str,
    questions: List[str],
    project_objective: str,
    model: str = llm.DEFAULT_MODEL,
) -> ExpertAnalysis:
    transcript_block = _transcript_block(transcript)
    questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    system = GROUNDING_RULES
    user = f"""Project objective: {project_objective}

Transcript for {display_name} (segment IDs are in square brackets at the start of each line):
---
{transcript_block}
---

Interview guide questions to answer, using ONLY this transcript:
{questions_block}

Return a JSON array with one object per question, in the same order, each with this shape:
{{
  "question": "<the question text>",
  "status": "answered" | "partial" | "not_discussed",
  "answer": "<your synthesized answer in your own words, 1-4 sentences. Empty string if not_discussed.>",
  "citations": [{{"segment_id": "E1-S014"}}, ...],
  "quotes": [{{"text": "<verbatim quote from the transcript>", "segment_id": "E1-S014"}}, ...]
}}

"quotes" is optional per question (use it when there's a genuinely useful, specific line -- don't force one). "citations" should list every segment_id your answer draws on."""

    data = llm.call_claude_json(client, system, user, model=model, max_tokens=4096)

    results: List[QuestionResult] = []
    for item in data:
        citations = verify.verify_citations(item.get("citations", []), transcript)
        quotes = verify.verify_and_flag_quotes(item.get("quotes", []), transcript)
        # attach timestamps for display
        for c in citations:
            seg = transcript.find_segment(c.get("segment_id", ""))
            c["timestamp"] = seg.timestamp if seg else None
            c["label"] = seg.citation_label() if seg else "[unknown segment]"
        for q in quotes:
            seg = transcript.find_segment(q.get("segment_id", ""))
            q["timestamp"] = seg.timestamp if seg else None
            q["label"] = seg.citation_label() if seg else "[unknown segment]"
        results.append(
            QuestionResult(
                question=item.get("question", ""),
                status=item.get("status", "not_discussed"),
                answer=item.get("answer", ""),
                citations=citations,
                quotes=quotes,
            )
        )
    return ExpertAnalysis(expert_label=transcript.expert_label, display_name=display_name, results=results)


def cross_expert_analysis(
    client: Groq,
    expert_analyses: List[ExpertAnalysis],
    questions: List[str],
    project_objective: str,
    model: str = llm.DEFAULT_MODEL,
) -> List[dict]:
    """
    For each interview-guide question, summarize common themes and
    disagreements across experts, citing back to (expert, segment_id).
    Operates on the already-grounded per-expert answers + citations rather
    than raw transcripts, so it can't drift from what was already verified.
    """
    per_expert_blocks = []
    for ea in expert_analyses:
        lines = [f"=== {ea.display_name} ({ea.expert_label}) ==="]
        for r in ea.results:
            cite_ids = ", ".join(c["segment_id"] for c in r.citations) or "none"
            lines.append(f"Q: {r.question}\nStatus: {r.status}\nAnswer: {r.answer}\nCitations: {cite_ids}")
            for q in r.quotes:
                mark = "OK" if q.get("verified") else "UNVERIFIED-SKIP"
                if q.get("verified"):
                    lines.append(f'  Quote [{q["segment_id"]}]: "{q["text"]}"')
        per_expert_blocks.append("\n".join(lines))
    all_blocks = "\n\n".join(per_expert_blocks)

    questions_block = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    system = GROUNDING_RULES + (
        "\nYou are now comparing already-extracted, per-expert answers (not raw transcripts). "
        "Only use segment_id values that literally appear in the material below -- never invent one."
    )
    user = f"""Project objective: {project_objective}

Below are structured answers already extracted from three expert transcripts, per interview-guide question.
---
{all_blocks}
---

Interview guide questions:
{questions_block}

For EACH question, identify:
- "themes": points where 2 or more experts agree or converge
- "disagreements": points where experts differ, give different emphasis, or contradict each other

Return a JSON array, one object per question:
{{
  "question": "<question text>",
  "themes": [
    {{"summary": "<the shared point, in your own words>", "supporting_experts": [{{"expert": "<display name>", "segment_id": "E1-S014"}}, ...]}}
  ],
  "disagreements": [
    {{"summary": "<what the disagreement is about, in your own words>", "positions": [{{"expert": "<display name>", "position": "<their view in a few words>", "segment_id": "E2-S009"}}, ...]}}
  ]
}}

If a question has no clear cross-expert theme or disagreement (e.g. only one expert addressed it), return an empty list for that field rather than forcing one. Do not compare experts on a question none of them addressed."""

    data = llm.call_claude_json(client, system, user, model=model, max_tokens=4096)
    return data


def answer_chat_question(
    client: Groq,
    question: str,
    transcripts: List[Transcript],
    chat_history: Optional[List[dict]] = None,
    model: str = llm.DEFAULT_MODEL,
) -> dict:
    """
    Free-form Q&A across all transcripts. Stuffs full transcripts into
    context (fine at n=3; see README for the RAG approach needed at scale).
    """
    blocks = []
    for t in transcripts:
        blocks.append(f"=== {t.expert_label} ===\n{_transcript_block(t)}")
    all_transcripts_block = "\n\n".join(blocks)

    history_block = ""
    if chat_history:
        turns = []
        for turn in chat_history[-6:]:  # keep recent context only
            turns.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
        history_block = "\n\nPrevious turns in this conversation (for context only):\n" + "\n\n".join(turns)

    system = GROUNDING_RULES + (
        "\nYou can draw on all three transcripts together to answer the user's question. "
        "If the transcripts don't contain the answer, say so plainly -- do not guess."
    )
    user = f"""All three transcripts:
---
{all_transcripts_block}
---
{history_block}

User question: {question}

Return a single JSON object:
{{
  "answer": "<your answer, in your own words, grounded only in the transcripts>",
  "citations": [{{"expert_label": "Expert 1", "segment_id": "E1-S014"}}, ...],
  "quotes": [{{"expert_label": "Expert 1", "text": "<verbatim quote>", "segment_id": "E1-S014"}}, ...],
  "answerable": true | false
}}

If none of the transcripts address the question, set "answerable" to false, explain that briefly in "answer", and leave citations/quotes empty."""

    data = llm.call_claude_json(client, system, user, model=model, max_tokens=2048)

    by_label = {t.expert_label: t for t in transcripts}
    citations = data.get("citations", [])
    for c in citations:
        t = by_label.get(c.get("expert_label", ""))
        seg = t.find_segment(c.get("segment_id", "")) if t else None
        c["exists"] = seg is not None
        c["timestamp"] = seg.timestamp if seg else None
        c["label"] = seg.citation_label() if seg else "[unknown segment]"

    quotes = data.get("quotes", [])
    verified_quotes = []
    for q in quotes:
        t = by_label.get(q.get("expert_label", ""))
        verified = verify.verify_quote(q.get("text", ""), q.get("segment_id", ""), t) if t else False
        seg = t.find_segment(q.get("segment_id", "")) if t else None
        verified_quotes.append(
            {
                **q,
                "verified": verified,
                "timestamp": seg.timestamp if seg else None,
                "label": seg.citation_label() if seg else "[unknown segment]",
            }
        )

    data["citations"] = citations
    data["quotes"] = verified_quotes
    return data
