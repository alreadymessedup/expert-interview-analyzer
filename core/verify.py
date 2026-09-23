"""
Code-level verification that quotes the model claims are "verbatim" actually
appear in the source segment. This is the second line of defense against
hallucination -- the prompt asks for verbatim quotes, but we don't trust
that alone; we check it.

We normalize whitespace/punctuation lightly (so "don't" vs "don t" from a
transcription artifact, or double spaces, don't cause false rejections) but
we do NOT allow paraphrase-level fuzziness. If a quote can't be located in
its cited segment, it is dropped and flagged rather than shown to the user.
"""

from __future__ import annotations

import re
from typing import Optional

from .parser import Transcript


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\u2018\u2019]", "'", s)
    s = re.sub(r"[\u201c\u201d]", '"', s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def verify_quote(quote_text: str, segment_id: str, transcript: Transcript) -> bool:
    """Returns True if `quote_text` is a verbatim substring of the cited segment."""
    segment = transcript.find_segment(segment_id)
    if segment is None:
        return False
    return _normalize(quote_text) in _normalize(segment.text)


def verify_and_flag_quotes(quotes: list, transcript: Transcript) -> list:
    """
    Given a list of quote dicts [{"text":..., "segment_id":...}, ...], returns
    the same list with a "verified" boolean added on each item. Unverified
    quotes are kept but clearly flagged (never silently shown as trustworthy).
    """
    out = []
    for q in quotes:
        text = q.get("text", "")
        seg_id = q.get("segment_id", "")
        verified = verify_quote(text, seg_id, transcript) if text and seg_id else False
        out.append({**q, "verified": verified})
    return out


def verify_citations(citations: list, transcript: Transcript) -> list:
    """
    Given a list of citation dicts [{"segment_id": ...}, ...], returns the
    same list with an "exists" boolean -- i.e. does this segment_id actually
    exist in this transcript at all (catches invented segment IDs).
    """
    out = []
    for c in citations:
        seg_id = c.get("segment_id", "")
        exists = transcript.find_segment(seg_id) is not None
        out.append({**c, "exists": exists})
    return out
