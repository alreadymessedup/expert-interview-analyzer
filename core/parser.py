"""
Transcript parsing.

Goal: turn whatever transcript file the user uploads into a list of
`Segment` objects that carry a stable, citable ID and (if available) a
timestamp. Everything downstream (LLM prompts, UI, quote verification)
cites back to these segments, so this is the foundation of traceability.

Supported inputs:
  - Plain text with inline timestamps, e.g.
        [00:12:34] Dr. Muller: Adoption in Germany is still limited...
        00:12:34 - Dr. Muller: Adoption in Germany is still limited...
        (00:12:34) Adoption in Germany is still limited...   (no speaker)
  - WebVTT / SRT exports (Zoom, Teams, Otter, etc.) using "-->" timestamp lines
  - Plain text with NO timestamps at all (paragraphs / lines). In this case
    segments are still numbered and citable, but the UI clearly shows that
    timestamps aren't available and quotes are only traceable to a segment
    number, not a time in the recording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    id: str  # e.g. "E1-S014" -- stable citation key
    index: int  # order within the transcript
    timestamp: Optional[str]  # "HH:MM:SS" or None
    speaker: Optional[str]
    text: str

    def citation_label(self) -> str:
        if self.timestamp:
            return f"[{self.timestamp}]"
        return f"[segment {self.index}]"


@dataclass
class Transcript:
    expert_label: str  # e.g. "Expert 1" or a real name the user supplies
    segments: List[Segment] = field(default_factory=list)
    has_timestamps: bool = False
    source_filename: str = ""

    def full_text(self) -> str:
        return "\n".join(s.text for s in self.segments)

    def find_segment(self, segment_id: str) -> Optional[Segment]:
        for s in self.segments:
            if s.id == segment_id:
                return s
        return None


_TS_TOKEN = r"(\d{1,2}:\d{2}(?::\d{2})?)"

# [00:12:34] Speaker: text   OR   (00:12:34) Speaker: text  OR  00:12:34 - Speaker: text
_INLINE_TS_RE = re.compile(
    rf"^\s*[\[\(]?{_TS_TOKEN}[\]\)]?\s*[-–—:]?\s*(?:(?P<speaker>[A-Za-z][\w .'\-]{{0,60}}):\s*)?(?P<text>.*)$"
)

# SRT/VTT cue timing line: 00:12:34,000 --> 00:12:40,000  or with dots
_CUE_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2})[.,]\d{1,3}\s*-->\s*(\d{1,2}:\d{2}:\d{2})[.,]\d{1,3}"
)

_SPEAKER_LINE_RE = re.compile(r"^(?P<speaker>[A-Za-z][\w .'\-]{0,60}):\s*(?P<text>.+)$")

# A line containing ONLY a timestamp, e.g. "00:00" or "01:12" or "[00:12:34]"
_STANDALONE_TS_RE = re.compile(rf"^\s*[\[\(]?{_TS_TOKEN}[\]\)]?\s*$")


def _normalize_ts(ts: str) -> str:
    parts = ts.split(":")
    if len(parts) == 2:
        return f"00:{int(parts[0]):02d}:{int(parts[1]):02d}"
    h, m, s = parts
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"


def _parse_srt_vtt(raw: str, expert_label: str, filename: str) -> Transcript:
    lines = raw.splitlines()
    segments: List[Segment] = []
    idx = 0
    i = 0
    current_speaker = None
    while i < len(lines):
        line = lines[i]
        m = _CUE_RE.search(line)
        if m:
            start_ts = _normalize_ts(m.group(1))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() != "" and not _CUE_RE.search(lines[i]):
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines).strip()
            # strip VTT position/style tags like <v Speaker>
            speaker = None
            vtag = re.match(r"^<v\s+([^>]+)>(.*)$", text)
            if vtag:
                speaker = vtag.group(1).strip()
                text = vtag.group(2).strip()
            else:
                sm = _SPEAKER_LINE_RE.match(text)
                if sm:
                    speaker = sm.group("speaker").strip()
                    text = sm.group("text").strip()
            if speaker:
                current_speaker = speaker
            text = re.sub(r"<[^>]+>", "", text).strip()  # strip other VTT tags
            if text:
                idx += 1
                segments.append(
                    Segment(
                        id=f"{expert_label}-S{idx:03d}",
                        index=idx,
                        timestamp=start_ts,
                        speaker=speaker or current_speaker,
                        text=text,
                    )
                )
        else:
            i += 1
    return Transcript(
        expert_label=expert_label,
        segments=segments,
        has_timestamps=True,
        source_filename=filename,
    )


def _parse_stacked_timestamps(raw: str, expert_label: str, filename: str) -> Optional[Transcript]:
    """
    Handles the pattern where a timestamp sits alone on its own line,
    followed (usually after a blank line) by a "Speaker: text" line, e.g.:

        00:00
        Interviewer: Thanks for joining...

        00:18
        Dr. Martin: Adoption is growing...

    Common output of manual transcription and several auto-transcription
    tools. Tried before the same-line inline parser because a timestamp
    line with nothing else on it would otherwise be silently skipped.
    """
    blocks = [b for b in re.split(r"\n\s*\n", raw.strip()) if b.strip()]
    if not blocks:
        return None

    segments: List[Segment] = []
    current_speaker = None
    idx = 0
    matched_blocks = 0

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        m = _STANDALONE_TS_RE.match(lines[0])
        if not m:
            continue  # e.g. the header block (name/role/market) before the first timestamp
        matched_blocks += 1
        ts = _normalize_ts(m.group(1))
        rest_text = " ".join(lines[1:]).strip()
        speaker = None
        sm = _SPEAKER_LINE_RE.match(rest_text)
        if sm:
            speaker = sm.group("speaker").strip()
            rest_text = sm.group("text").strip()
            current_speaker = speaker
        if not rest_text:
            continue
        idx += 1
        segments.append(
            Segment(
                id=f"{expert_label}-S{idx:03d}",
                index=idx,
                timestamp=ts,
                speaker=speaker or current_speaker,
                text=rest_text,
            )
        )

    if segments and matched_blocks >= max(2, len(blocks) * 0.3):
        return Transcript(
            expert_label=expert_label,
            segments=segments,
            has_timestamps=True,
            source_filename=filename,
        )
    return None


def _parse_inline_timestamps(raw: str, expert_label: str, filename: str) -> Optional[Transcript]:
    lines = [l for l in raw.splitlines() if l.strip()]
    if not lines:
        return None
    matched = 0
    segments: List[Segment] = []
    current_speaker = None
    idx = 0
    for line in lines:
        m = _INLINE_TS_RE.match(line)
        if m and m.group(1):
            matched += 1
            ts = _normalize_ts(m.group(1))
            speaker = m.group("speaker")
            text = m.group("text").strip()
            if speaker:
                current_speaker = speaker.strip()
            if not text:
                continue
            idx += 1
            segments.append(
                Segment(
                    id=f"{expert_label}-S{idx:03d}",
                    index=idx,
                    timestamp=ts,
                    speaker=speaker.strip() if speaker else current_speaker,
                    text=text,
                )
            )
    # require a reasonable fraction of lines to carry timestamps to trust this parse
    if matched >= max(3, len(lines) * 0.3):
        return Transcript(
            expert_label=expert_label,
            segments=segments,
            has_timestamps=True,
            source_filename=filename,
        )
    return None


def _parse_plain_text(raw: str, expert_label: str, filename: str) -> Transcript:
    # Fall back: split into paragraphs (blank-line separated), else sentences-ish chunks.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraphs) < 2:
        # split by single newlines instead
        paragraphs = [p.strip() for p in raw.splitlines() if p.strip()]

    segments: List[Segment] = []
    current_speaker = None
    idx = 0
    for para in paragraphs:
        speaker = None
        text = para
        sm = _SPEAKER_LINE_RE.match(para)
        if sm:
            speaker = sm.group("speaker").strip()
            text = sm.group("text").strip()
            current_speaker = speaker
        if not text:
            continue
        idx += 1
        segments.append(
            Segment(
                id=f"{expert_label}-S{idx:03d}",
                index=idx,
                timestamp=None,
                speaker=speaker or current_speaker,
                text=text,
            )
        )
    return Transcript(
        expert_label=expert_label,
        segments=segments,
        has_timestamps=False,
        source_filename=filename,
    )


def parse_transcript(raw: str, expert_label: str, filename: str = "") -> Transcript:
    """
    Best-effort parser that tries, in order:
      1. SRT/VTT cue-based parsing (if '-->' timing lines are present)
      2. Inline per-line timestamp parsing (e.g. "[00:12:34] Name: text")
      3. Plain-text paragraph fallback (no timestamps -> segment-number citations only)
    """
    if not raw or not raw.strip():
        return Transcript(expert_label=expert_label, segments=[], has_timestamps=False, source_filename=filename)

    if _CUE_RE.search(raw):
        t = _parse_srt_vtt(raw, expert_label, filename)
        if t.segments:
            return t

    t = _parse_stacked_timestamps(raw, expert_label, filename)
    if t is not None and t.segments:
        return t

    t = _parse_inline_timestamps(raw, expert_label, filename)
    if t is not None and t.segments:
        return t

    return _parse_plain_text(raw, expert_label, filename)
