# Expert Interview Analyzer

A small app that analyzes 3 expert-call transcripts from the same market research
project: answers the interview-guide questions per expert, extracts verbatim
quotes, cites a timestamp/segment for every answer, compares themes and
disagreements across experts, and lets you ask free-form questions across all
three transcripts — with every claim traceable back to source.

Built around the "European Robotic Surgery Market" interview guide, but the
guide (objective + questions) is editable in the sidebar, so it works for any
project with the same shape.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

add .env             
# then edit .env and set GROQ_API_KEY=gsk_... (get one at console.groq.com/keys)

uvicorn server:app --reload
```

Open **http://localhost:8000** — FastAPI serves the API under `/api/*` and
the static frontend (`frontend/index.html` + `frontend/app.js`, plain
HTML/CSS/JS with Tailwind loaded from a CDN, no build step) at `/`.

**A `.env` file is the recommended way to set the API key** — `server.py`
loads it automatically (`python-dotenv`) regardless of OS or which shell
you use. Setting the key as a real environment variable instead also
works, but the command differs by shell and it must be set in the *same*
terminal session that then launches `uvicorn` (child processes only
inherit variables already present when they start):

| Shell | Command |
|---|---|
| bash / zsh (Linux, macOS) | `export GROQ_API_KEY=gsk_...` |
| Windows Command Prompt | `set GROQ_API_KEY=gsk_...` |
| Windows PowerShell | `$env:GROQ_API_KEY="gsk_..."` |

If neither is set, the app falls back to whatever you paste into the
**Groq API key** field in the sidebar at runtime — that value is sent with
each request and never touches the server's environment.

## Using it

1. Upload three transcript files (`.txt`, `.vtt`, or `.srt`) — one per
   expert — and optionally rename them.
2. Adjust the project objective / interview-guide questions in the sidebar
   if needed.
3. Click **Analyze transcripts**. This calls `POST /api/analyze`, which runs
   the full pipeline server-side and returns an `analysis_id` the frontend
   holds onto for the rest of the session (see "Session state" below).
4. Read results in the per-expert tabs, the **Cross-expert analysis** tab,
   or ask your own questions in the **Ask a question** tab (`POST
   /api/chat`). **Raw transcripts** is there so you can always double-check
   a citation by eye.


## Architecture

```
transcript file(s)  (uploaded via multipart/form-data to POST /api/analyze)
      │
      ▼
 parser.py          → Segment objects: stable ID + timestamp + speaker + text
      │
      ▼
 pipeline.py ─┬─ answer_questions_for_expert()  → per-expert, per-question JSON
              │      (grounded prompt, forces citations + status, one call per expert)
              │
              ├─ cross_expert_analysis()         → themes & disagreements
              │      (operates on the already-grounded per-expert answers,
              │       not raw transcripts, so it can't drift further from source)
              │
              └─ answer_chat_question()          → free-form Q&A across all 3
                     (full transcripts stuffed into context; see "scaling" below)
      │
      ▼
 verify.py          → code-level check: does every "verbatim" quote actually
                       appear in its cited segment? Unverified quotes are
                       dropped, not shown.
      │
      ▼
 server.py (FastAPI) → JSON API: POST /api/analyze, POST /api/chat,
                        GET /api/analysis/{id}, GET /api/default-guide,
                        GET /api/models
      │
      ▼
 frontend/           → vanilla JS + Tailwind (CDN), no build step or framework.
                        app.js renders per-expert tabs, cross-analysis, chat,
                        and a raw-transcript viewer by fetching the API above.
```

`llm.py` is a thin wrapper around the Groq API (OpenAI-compatible chat
completions endpoint, `https://api.groq.com/openai/v1`; model selection,
JSON-mode parsing with fence-stripping and a one-shot retry).

### Persistence

HTTP is stateless, so every analysis needs somewhere durable to live
between requests — and ideally between server restarts. `server.py` has no
in-process session state at all; every request reads from and writes to
`core/store.py`, a persistence layer backed by **ChromaDB**'s
`PersistentClient`, which stores everything as local files under
`CHROMA_DIR` (`./chroma_data` by default) — no separate database process to
run, and analyses survive a restart.

Four Chroma collections cover a full analysis:

| Collection | One row per | Used for |
|---|---|---|
| `analyses` | analysis | guide, model, cross-analysis, expert names |
| `segments` | transcript segment | the citation source every answer/quote points back to |
| `expert_answers` | (expert, question) result | per-expert Q&A, with citations/quotes JSON-encoded |
| `chat_turns` | chat message | full chat history, in order |

`GET /api/analyses` lists everything saved so far (newest first), and the
frontend surfaces this as a **"Previous analyses"** panel with **Resume**
and **Delete** buttons — resuming loads a past analysis straight from disk
with zero re-uploading and zero LLM calls.

Chroma is used here purely as a durable, filterable store — not yet for
semantic search, so every write supplies a placeholder embedding instead of
pulling in a real embedding model, keeping this dependency-light and fully
offline. The `segments` collection's schema is deliberately already shaped
for the "retrieval instead of context-stuffing" scaling item below: adding
real embeddings later to power search only touches `core/store.py`, nothing
about how segments are stored changes.



## How citations/timestamps are handled

1. The parser assigns every transcript line a **stable segment ID**
   (`E1-S014`) plus its timestamp (or `None` if the source has none).
2. Every prompt sent to the model shows the transcript as
   `[segment_id] [timestamp] Speaker: text` per line, and the model is
   instructed to cite `segment_id` values for every claim — never a
   timestamp it invents from memory.
3. On the way back, the app looks up each cited `segment_id` in the parsed
   transcript to render the actual timestamp/label. If the model returns a
   `segment_id` that doesn't exist, that's caught and flagged (`exists:
   False`) rather than silently trusted.

## How hallucinations are reduced

This is a layered approach — prompting alone is not treated as sufficient:

1. **Closed-book grounding in the prompt.** The system prompt explicitly
   forbids outside knowledge and inference beyond what's stated, and makes
   "not discussed" an expected, valid answer rather than something the
   model has to work around.
2. **Forced citations.** The output schema requires a `segment_id` for
   every answer and every quote. A claim with no citation is a schema
   violation, not just a style nit — this catches most drift on its own.
3. **Code-level quote verification (`verify.py`).** The model is told
   quotes must be verbatim, but that instruction is not trusted blindly:
   every returned quote is checked as an exact (whitespace/quote-mark
   normalized) substring of its cited segment's actual text. Anything that
   doesn't match is dropped before the user ever sees it, not just
   flagged after the fact.
4. **Segment-ID existence checks.** Every cited `segment_id` is checked
   against the real transcript; invented IDs are marked `exists: False`
   instead of rendered as if they were valid.
5. **Temperature 0** for all extraction/comparison calls — this is a
   grounding task, not a creative one.
6. **Cross-expert analysis reasons over already-grounded, already-cited
   per-expert answers**, not raw transcripts a second time — so it can't
   introduce a new, unverified claim about what an expert said; it can only
   compare claims that already passed the per-expert grounding step.
7. **Raw transcript viewer in the UI.** Every answer is one click away from
   the actual transcript text, so a human can always spot-check — this
   product deliberately keeps a human in the loop rather than presenting
   itself as fully autonomous.

What this does *not* do: fact-check what the expert said against outside
reality. It only guarantees the app's summary is faithful to what's *in the
transcript* — accuracy of the expert's own claims is out of scope by
design, per "do not invent information."

## Scaling from 3 transcripts to 30+

The current design intentionally keeps things simple at n=3 by stuffing
full transcripts into context. That stops being efficient/reliable well
before n=30. Changes for scale:

1. **Batch ingestion, decoupled from the request/response cycle.**
   Persistence itself is already solved (Chroma, see above); what's still
   synchronous is *ingestion* — `/api/analyze` runs all per-expert LLM calls
   inline before responding. At scale, move that to a background
   job/worker queue (the same per-expert call as today, just async and
   parallel — Groq's inference speed helps a lot here) that writes into the
   same `core/store.py` as it finishes each expert, so the UI doesn't block
   on a single long-lived request for 30 transcripts.
2. **Real embeddings + retrieval instead of context-stuffing for
   chat/cross-analysis.** The `segments` Chroma collection already stores
   every segment with its full citation metadata (expert, segment ID,
   timestamp) — it just uses placeholder embeddings today. Swapping those
   for real ones (any embeddings model) turns it into an actual retrieval
   index: for the "ask a question across transcripts" feature, retrieve
   the top-k most relevant segments across all 30+ transcripts rather than
   pasting everything into one prompt. This is a change entirely inside
   `core/store.py` — nothing about the schema or the citation mechanism
   changes.
3. **Two-stage cross-analysis.** A single prompt comparing 30 experts at
   once won't fit context and will produce mushy summaries. Instead:
   cluster per-expert answers/quotes by embedding similarity per question,
   then run the "identify themes & disagreements" prompt per cluster, then
   a final pass that merges cluster-level summaries. Keeps each individual
   LLM call focused and grounded.
4. **Move from Chroma's local file store to a multi-writer database if
   this ever needs concurrent multi-user access.** Chroma's
   `PersistentClient` is single-process and file-backed — ideal for a local
   single-user tool, but it doesn't give concurrent writers or network
   access the way a real database server does. At real multi-user scale,
   the natural next step is Postgres with the `pgvector` extension
   (transcripts, segments, extracted answers, and real embeddings all in
   one place), keeping the same schema `core/store.py` already defines.
5. **Systematic citation auditing at scale.** The current per-quote
   verification stays essential, but at 30+ transcripts it's also worth
   periodically sampling a % of generated answers for manual QA, since
   silent prompt drift is harder to eyeball at that volume.

## Known limitations

- Speaker/timestamp parsing is regex-based and tuned for common transcript
  export formats; unusual formats may fall back to "no timestamps" mode
  (still fully citable by segment number, just not by clock time).
- The chat feature re-sends full transcripts on every turn (no real
  retrieval index yet, even though persistence is in place) — fine at n=3,
  called out above as the first thing to change at scale.
- No authentication/multi-user support — this is a local single-user tool
  as scoped, and Chroma's `PersistentClient` is single-process/file-backed
  rather than a networked database.
- To wipe all saved data, stop the server and delete the `chroma_data/`
  directory (or whatever `CHROMA_DIR` points to); there's no in-app "clear
  everything" button.
