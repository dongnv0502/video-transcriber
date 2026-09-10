"""Deterministic rules, case normalization, and glossary loading."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from transcriber.paths import fold


class GlossaryFile(BaseModel):
    version: int
    terms: list[str]


class CorrectionRule(BaseModel):
    from_: str = Field(alias="from")
    to: str
    match: Literal["literal", "regex"] = "literal"
    auto: bool = True
    requires_context: list[str] = Field(default_factory=list)


class CorrectionsConfigFile(BaseModel):
    version: int
    corrections: list[CorrectionRule]


@dataclass
class RuleHit:
    rule_from: str
    rule_to: str
    reason: str


def load_glossary(path: Path | str) -> GlossaryFile:
    """Load and parse glossary YAML file."""
    p = Path(path).resolve()
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return GlossaryFile.model_validate(data)


def load_corrections(path: Path | str) -> CorrectionsConfigFile:
    """Load and parse corrections YAML file."""
    p = Path(path).resolve()
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CorrectionsConfigFile.model_validate(data)


def apply_case_normalization(text: str, terms: list[str]) -> tuple[str, list[str]]:
    """Normalize casing of terms matching glossary, without changing words or letters.

    Matches case-insensitively and only replaces if matched text differs from canonical term
    by letter case alone.
    """
    applied: list[str] = []
    # Sort terms by length descending to match compound terms before single words
    sorted_terms = sorted(terms, key=len, reverse=True)

    result_text = text
    for term in sorted_terms:
        # Build regex anchored to non-word characters, allowing variable whitespace between words
        escaped_words = [re.escape(w) for w in term.split()]
        pattern_str = r"(?<![\w])" + r"\s+".join(escaped_words) + r"(?![\w])"
        pattern = re.compile(pattern_str, re.IGNORECASE)

        def replacer(match: re.Match[str]) -> str:
            matched_span = match.group(0)
            # Only replace if identical casefolded but different in actual casing
            if matched_span.casefold() == term.casefold() and matched_span != term:
                applied.append(term)
                return term
            return matched_span

        result_text = pattern.sub(replacer, result_text)

    return result_text, applied


def apply_auto_rules(
    prev: str,
    text: str,
    next: str,
    corrections: list[CorrectionRule],
) -> tuple[str, list[RuleHit]]:
    """Apply auto=True corrections whose context conditions are met across 3-segment window."""
    applied: list[RuleHit] = []
    context_window = fold(f"{prev} {text} {next}")

    result_text = text
    for rule in corrections:
        if not rule.auto:
            continue

        # Check context requirement: if specified, at least one keyword must be present in folded window
        if rule.requires_context:
            matched_context = any(fold(ctx) in context_window for ctx in rule.requires_context)
            if not matched_context:
                continue

        # Match pattern
        if rule.match == "literal":
            words = [re.escape(w) for w in rule.from_.split()]
            pattern_str = r"(?<![\w])" + r"\s+".join(words) + r"(?![\w])"
            pattern = re.compile(pattern_str, re.IGNORECASE)
        else:
            pattern = re.compile(rule.from_, re.IGNORECASE)

        if pattern.search(result_text):
            result_text = pattern.sub(rule.to, result_text)
            hit = RuleHit(
                rule_from=rule.from_,
                rule_to=rule.to,
                reason=f"Rule match: '{rule.from_}' -> '{rule.to}'",
            )
            applied.append(hit)

    return result_text, applied


def candidate_hits(
    prev: str,
    text: str,
    next: str,
    corrections: list[CorrectionRule],
) -> list[str]:
    """Identify corrections that match pattern but are NOT auto-applied.

    Matches that are candidate-only (auto=False) or have unsatisfied context.
    """
    candidates: list[str] = []
    context_window = fold(f"{prev} {text} {next}")

    for rule in corrections:
        # Match pattern
        if rule.match == "literal":
            words = [re.escape(w) for w in rule.from_.split()]
            pattern_str = r"(?<![\w])" + r"\s+".join(words) + r"(?![\w])"
            pattern = re.compile(pattern_str, re.IGNORECASE)
        else:
            pattern = re.compile(rule.from_, re.IGNORECASE)

        if pattern.search(text):
            if not rule.auto:
                candidates.append(f"Candidate (manual): '{rule.from_}' -> '{rule.to}'")
            else:
                # auto=True but check if context was unsatisfied
                if rule.requires_context:
                    matched_context = any(fold(ctx) in context_window for ctx in rule.requires_context)
                    if not matched_context:
                        candidates.append(
                            f"Candidate (unmet context): '{rule.from_}' -> '{rule.to}'"
                        )

    return candidates
