from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdflib import Graph, Literal


RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SURVEY_NS = (
    "http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/"
)

# Focus on practical business literals likely to appear in user questions.
TARGET_PREDICATE_URIS = {
    RDFS_LABEL,
    SURVEY_NS + "nameplateLabel",
    SURVEY_NS + "regionName",
    SURVEY_NS + "technologyCategoryName",
    SURVEY_NS + "companyName",
    SURVEY_NS + "periodLabel",
    SURVEY_NS + "baselineType",
}
TARGET_PREDICATE_SUFFIXES = {
    "label",
    "nameplateLabel",
    "regionName",
    "technologyCategoryName",
    "companyName",
    "periodLabel",
    "baselineType",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9#/_\.-]+")
ALNUM_RE = re.compile(r"[a-z0-9]+")
MULTISPACE_RE = re.compile(r"\s+")


def normalize_alias(text: str) -> str:
    if not text:
        return ""
    return "".join(ALNUM_RE.findall(text.lower()))


def _digit_signature(norm: str) -> str:
    return "".join(ch for ch in norm if ch.isdigit())


def _prefix_key(norm: str) -> str:
    if len(norm) >= 3:
        return norm[:3]
    return norm


def _predicate_is_candidate(uri: str) -> bool:
    if uri in TARGET_PREDICATE_URIS:
        return True
    suffix = uri.rsplit("/", 1)[-1]
    return suffix in TARGET_PREDICATE_SUFFIXES


@dataclass
class EntityAliasIndex:
    # normalized alias -> canonical-label frequency counter
    key_to_labels: Dict[str, Counter]
    # quick retrieval buckets for fuzzy matching
    keys_by_digit_signature: Dict[str, List[str]]
    keys_by_prefix: Dict[str, List[str]]

    def best_label(self, norm_key: str) -> Optional[str]:
        counter = self.key_to_labels.get(norm_key)
        if not counter:
            return None
        # Prefer higher frequency, then shorter canonical label.
        items = sorted(counter.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
        return items[0][0] if items else None


@dataclass
class QuestionCanonicalization:
    original_question: str
    effective_question: str
    mappings: List[Dict[str, Any]]

    @property
    def changed(self) -> bool:
        return self.original_question.strip() != self.effective_question.strip()


def build_entity_alias_index(
    graph: Graph,
    max_literals: int = 600_000,
    min_key_len: int = 2,
    max_key_len: int = 64,
) -> EntityAliasIndex:
    key_to_labels: Dict[str, Counter] = defaultdict(Counter)
    seen_literals = 0

    for _, p, o in graph:
        if not isinstance(o, Literal):
            continue
        p_uri = str(p)
        if not _predicate_is_candidate(p_uri):
            continue

        label = str(o).strip()
        if not label:
            continue
        norm = normalize_alias(label)
        if len(norm) < min_key_len or len(norm) > max_key_len:
            continue

        key_to_labels[norm][label] += 1
        seen_literals += 1
        if seen_literals >= max_literals:
            break

    keys_by_digit_signature: Dict[str, List[str]] = defaultdict(list)
    keys_by_prefix: Dict[str, List[str]] = defaultdict(list)
    for key in key_to_labels.keys():
        dig = _digit_signature(key)
        if dig:
            keys_by_digit_signature[dig].append(key)
        keys_by_prefix[_prefix_key(key)].append(key)

    return EntityAliasIndex(
        key_to_labels=dict(key_to_labels),
        keys_by_digit_signature=dict(keys_by_digit_signature),
        keys_by_prefix=dict(keys_by_prefix),
    )


def _candidate_spans(question: str, max_ngram: int = 4) -> List[Tuple[int, int, str]]:
    tokens = [(m.start(), m.end(), m.group(0)) for m in TOKEN_RE.finditer(question)]
    spans: List[Tuple[int, int, str]] = []
    if not tokens:
        return spans

    n = len(tokens)
    for i in range(n):
        for k in range(1, max_ngram + 1):
            j = i + k - 1
            if j >= n:
                break
            start = tokens[i][0]
            end = tokens[j][1]
            mention = question[start:end]
            spans.append((start, end, mention))
    return spans


def _fuzzy_lookup(norm: str, index: EntityAliasIndex) -> Tuple[Optional[str], float]:
    if not norm:
        return None, 0.0

    candidates: List[str] = []
    dig = _digit_signature(norm)
    if dig and dig in index.keys_by_digit_signature:
        candidates.extend(index.keys_by_digit_signature[dig])
    prefix = _prefix_key(norm)
    if prefix in index.keys_by_prefix:
        candidates.extend(index.keys_by_prefix[prefix])

    if not candidates:
        return None, 0.0

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)

    # Guardrail for very common prefixes.
    if len(deduped) > 4000:
        deduped = deduped[:4000]

    best_key = None
    best_ratio = 0.0
    for key in deduped:
        ratio = difflib.SequenceMatcher(None, norm, key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = key

    if best_key is None:
        return None, 0.0
    if best_ratio < 0.90:
        return None, best_ratio
    return index.best_label(best_key), best_ratio


def resolve_question_entities(
    question: str,
    index: EntityAliasIndex,
    max_matches: int = 5,
) -> List[Dict[str, object]]:
    if not question or not index.key_to_labels:
        return []

    candidates = []
    for start, end, mention in _candidate_spans(question, max_ngram=4):
        norm = normalize_alias(mention)
        if len(norm) < 2:
            continue

        canonical = index.best_label(norm)
        score = 0.0
        method = ""

        if canonical:
            score = 1.0
            method = "exact"
        else:
            # Fuzzy is only useful for entity-like mentions.
            # Require some specificity to avoid overlinking common words.
            has_digit = any(ch.isdigit() for ch in norm)
            if len(norm) < 5 and not has_digit:
                continue
            canonical, ratio = _fuzzy_lookup(norm, index=index)
            if not canonical:
                continue
            score = 0.60 + 0.40 * ratio
            method = "fuzzy"

        if not canonical:
            continue

        # Skip no-op matches.
        if normalize_alias(canonical) == norm and mention.strip().lower() == canonical.strip().lower():
            continue

        candidates.append(
            {
                "start": start,
                "end": end,
                "mention": mention,
                "canonical": canonical,
                "score": round(float(score), 4),
                "method": method,
            }
        )

    if not candidates:
        return []

    # Keep highest-confidence non-overlapping spans.
    candidates.sort(
        key=lambda x: (-float(x["score"]), -(int(x["end"]) - int(x["start"])), int(x["start"]))
    )
    chosen = []
    occupied: List[Tuple[int, int]] = []
    for c in candidates:
        s = int(c["start"])
        e = int(c["end"])
        overlap = any(not (e <= os or s >= oe) for os, oe in occupied)
        if overlap:
            continue
        chosen.append(c)
        occupied.append((s, e))
        if len(chosen) >= max_matches:
            break

    chosen.sort(key=lambda x: int(x["start"]))
    return chosen


def canonicalize_question(
    question: str,
    resolved_entities: Sequence[Dict[str, object]],
) -> str:
    if not question or not resolved_entities:
        return question

    out = question
    # Replace right-to-left to keep span offsets stable.
    for item in sorted(resolved_entities, key=lambda x: int(x["start"]), reverse=True):
        s = int(item["start"])
        e = int(item["end"])
        canonical = str(item.get("canonical", "")).strip()
        if not canonical:
            continue
        mention = out[s:e]
        if mention.strip().lower() == canonical.strip().lower():
            # No semantic gain from replacement.
            continue
        out = out[:s] + canonical + out[e:]
    out = MULTISPACE_RE.sub(" ", out).strip()
    return out


def canonicalize_question_with_index(
    question: str,
    index: Optional[EntityAliasIndex],
    max_matches: int = 5,
) -> QuestionCanonicalization:
    text = (question or "").strip()
    if not text or index is None:
        return QuestionCanonicalization(
            original_question=text,
            effective_question=text,
            mappings=[],
        )

    mappings = resolve_question_entities(text, index=index, max_matches=max_matches)
    effective = canonicalize_question(text, mappings)
    return QuestionCanonicalization(
        original_question=text,
        effective_question=effective,
        mappings=mappings,
    )
