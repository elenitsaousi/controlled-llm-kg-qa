"""Lightweight, additive slot-extraction helpers for KGQA questions.

Pure functions only: no Streamlit, no SPARQL execution, no dependency on
`app.py`, `llm.candidate_generation`, or the ranking pipeline. This module
does not replace or restructure candidate generation or ranking — it exists
so any layer can optionally consult a small set of well-tested extraction
helpers (as hints or guards) without risking a circular import or a change
to existing deterministic behavior. Nothing in the existing pipeline is
required to call this module; callers opt in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Set


REGION_LITERALS = {
    "americas": "americas",
    "europe": "europe",
    "japan": "japan",
    "china": "china",
    "all_other": "all other",
    "asia_pacific": "asia pacific",
}

_REGION_ALIASES = {
    "americas": ("americas", "america"),
    "europe": ("europe",),
    "japan": ("japan",),
    "china": ("china",),
    "all_other": ("all other",),
    "asia_pacific": ("asia pacific",),
}


def extract_region_scope(question: str) -> Set[str]:
    """Return the set of canonical region keys (see REGION_LITERALS) mentioned
    in the question, e.g. {"europe", "americas"} for "Europe and America combined".
    """
    q = (question or "").lower()
    found: Set[str] = set()
    for canonical, aliases in _REGION_ALIASES.items():
        if any(alias in q for alias in aliases):
            found.add(canonical)
    return found


@dataclass
class TimeWindow:
    kind: Optional[str] = None  # "relative_months" | "relative_quarters" | "relative_years"
    n: Optional[int] = None
    direction: Optional[str] = None  # "past" | "future"
    raw: str = ""

    @property
    def is_present(self) -> bool:
        return self.kind is not None


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_NUMBER_GROUP = r"(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_UNIT_GROUP = r"(?P<unit>months?|quarters?|years?)"

_PAST_TIME_RE = re.compile(
    rf"\b(?:past|last|recent|previous)\s+(?:the\s+)?{_NUMBER_GROUP}?\s*{_UNIT_GROUP}\b"
)
_FUTURE_TIME_RE = re.compile(
    rf"\b(?:next|upcoming|first)\s+(?:the\s+)?{_NUMBER_GROUP}?\s*{_UNIT_GROUP}\b"
)


def _kind_for_unit(unit: str) -> str:
    if unit.startswith("month"):
        return "relative_months"
    if unit.startswith("quarter"):
        return "relative_quarters"
    return "relative_years"


def _resolve_n(raw_n: str) -> Optional[int]:
    raw_n = (raw_n or "").strip()
    if not raw_n:
        return None
    if raw_n.isdigit():
        return int(raw_n)
    return _NUMBER_WORDS.get(raw_n)


def extract_time_window(question: str) -> TimeWindow:
    """Extract a relative time window from phrasing such as "last N months",
    "past year", "next 2 quarters", or "first upcoming quarter".

    Returns a TimeWindow with kind=None when no relative time phrase is
    present. `n` is None when the phrase has no explicit count (e.g. "last
    months", "the upcoming quarter") — callers decide their own default.
    """
    q = (question or "").lower()
    match = _PAST_TIME_RE.search(q)
    direction = "past"
    if not match:
        match = _FUTURE_TIME_RE.search(q)
        direction = "future"
    if not match:
        return TimeWindow()
    return TimeWindow(
        kind=_kind_for_unit(match.group("unit")),
        n=_resolve_n(match.group("n")),
        direction=direction,
        raw=match.group(0),
    )
