"""Suspicious segment detection based on ASR metrics, repetition, and glossary near-misses."""

import re
from rapidfuzz import fuzz

from transcriber.config import Settings
from transcriber.models import RawSegment
from transcriber.paths import fold
from transcriber.stages.rules import CorrectionRule, candidate_hits

# Flag string constants
LOW_LOGPROB = "LOW_LOGPROB"
HIGH_NOSPEECH = "HIGH_NOSPEECH"
HIGH_COMPRESSION = "HIGH_COMPRESSION"
REPETITION = "REPETITION"
FAST_TEXT = "FAST_TEXT"
SHORT_SEGMENT = "SHORT_SEGMENT"
GLOSSARY_NEAR_MISS = "GLOSSARY_NEAR_MISS"
CORRECTION_CANDIDATE = "CORRECTION_CANDIDATE"

REPETITION_REGEX = re.compile(r"(?i)\b(\w+(?:\s+\w+)?)\b(?:\s+\1\b){3,}")

def check_glossary_near_miss(folded_text: str, folded_terms: dict[str, str]) -> bool:
    """Check if any 1-3 token n-gram in folded text is a near-miss (ratio 78..99) to a glossary term.

    Ignores n-grams that are exactly equal to any folded term.
    """
    tokens = folded_text.split()
    n_tokens = len(tokens)
    if n_tokens == 0:
        return False

    for n in (1, 2, 3):
        for i in range(n_tokens - n + 1):
            ngram = " ".join(tokens[i : i + n])
            # If exact match to a term, it's not a near-miss
            if ngram in folded_terms:
                continue

            for term_folded in folded_terms:
                # Length filter to avoid expensive ratio comparisons on vastly mismatched strings
                if abs(len(ngram) - len(term_folded)) > max(len(term_folded), 4):
                    continue
                score = fuzz.ratio(ngram, term_folded)
                if 78 <= score < 100:
                    return True
    return False


def flag_segment(
    seg: RawSegment,
    prev: str,
    next: str,
    glossary_terms: list[str],
    corrections: list[CorrectionRule],
    settings: Settings | None = None,
) -> list[str]:
    """Inspect segment against acoustic metrics, repetition, and glossary heuristics.

    Returns a list of detected flag names.
    """
    flags: list[str] = []
    text = seg.text.strip()
    duration = max(seg.end - seg.start, 0.001)

    # 1. Acoustic / Whisper metrics
    if seg.avg_logprob is not None and seg.avg_logprob < -0.70:
        flags.append(LOW_LOGPROB)

    if seg.no_speech_prob is not None and seg.no_speech_prob > 0.60:
        flags.append(HIGH_NOSPEECH)

    if seg.compression_ratio is not None and seg.compression_ratio > 2.4:
        flags.append(HIGH_COMPRESSION)

    # 2. Text anomalies
    if REPETITION_REGEX.search(text):
        flags.append(REPETITION)

    if len(text) / duration > 25.0:
        flags.append(FAST_TEXT)

    if duration < 0.35 and len(text) > 12:
        flags.append(SHORT_SEGMENT)

    # 3. Glossary near-miss (phonetic / typo anomaly)
    if glossary_terms:
        folded_terms = {fold(t): t for t in glossary_terms}
        folded_text = fold(text)
        if check_glossary_near_miss(folded_text, folded_terms):
            flags.append(GLOSSARY_NEAR_MISS)

    # 4. Correction rule candidates (manual or unmet context)
    candidates = candidate_hits(prev, text, next, corrections)
    if candidates:
        flags.append(CORRECTION_CANDIDATE)

    return flags
