"""Stage: LLM-assisted and deterministic transcript correction with validation gate."""

import hashlib
import json
import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rapidfuzz import fuzz

from transcriber.config import Settings
from transcriber.db import Status, set_status, utc_now
from transcriber.llm.client import LlmClient, LlmError, LlmUnavailable
from transcriber.llm.prompt import (
    CORRECTION_SCHEMA,
    SYSTEM_PROMPT,
    build_user_message,
)
from transcriber.models import (
    CorrectedSegment,
    CorrectedTranscript,
    CorrectionDiffItem,
    CorrectionInfo,
    CorrectionMeta,
    CorrectionsFile,
    RawTranscript,
)
from transcriber.paths import atomic_write_json
from transcriber.stages import StageError
from transcriber.stages.rules import (
    apply_auto_rules,
    apply_case_normalization,
    load_corrections,
    load_glossary,
)
from transcriber.stages.suspicious import flag_segment

logger = logging.getLogger("transcriber.correct")


def validate_correction_item(
    raw_text: str,
    item: dict[str, Any],
) -> tuple[bool, str | None]:
    """Apply the 9-rule validation gate to an LLM correction item.

    Returns (accepted: bool, rejection_reason: str | None).
    """
    changed = item.get("changed", False)
    corrected = item.get("corrected")

    if not changed:
        return False, None

    if corrected is None or not str(corrected).strip():
        return False, "empty_correction"

    corrected_str = str(corrected).strip()
    raw_str = raw_text.strip()

    if corrected_str == raw_str:
        return False, "identical_to_raw"

    confidence = float(item.get("confidence", 0.0))
    if confidence < 0.85:
        return False, "confidence_below_0.85"

    ratio = fuzz.ratio(raw_str, corrected_str)
    if ratio < 60:
        return False, "fuzz_ratio_below_60"

    len_ratio = len(corrected_str) / max(len(raw_str), 1)
    if len_ratio < 0.7 or len_ratio > 1.4:
        return False, "length_ratio_outside_0.7_1.4"

    raw_numbers = re.findall(r"\d+", raw_str)
    corrected_numbers = re.findall(r"\d+", corrected_str)
    if raw_numbers != corrected_numbers:
        return False, "numbers_modified"

    raw_words = raw_str.split()
    corrected_words = corrected_str.split()
    if abs(len(corrected_words) - len(raw_words)) > 3:
        return False, "word_count_diff_exceeds_3"

    return True, None


def correct_video(
    conn: sqlite3.Connection,
    settings: Settings,
    row: sqlite3.Row,
    dry_run: bool = False,
    force: bool = False,
    client: LlmClient | None = None,
) -> CorrectedTranscript:
    """Execute deterministic and LLM-assisted correction on raw transcript."""
    slug = row["slug"]
    video_id = row["id"]
    out_video_dir = settings.output_dir.resolve() / slug
    raw_json_path = out_video_dir / "raw.json"
    corrected_json_path = out_video_dir / "corrected.json"
    corrections_json_path = out_video_dir / "corrections.json"

    # Step 1: Load raw transcript
    if not raw_json_path.exists():
        msg = f"raw.json not found in {out_video_dir}"
        logger.error(msg)
        set_status(conn, video_id, Status.FAILED, failed_stage="correct", error=msg, stage="correct")
        raise StageError("correct", msg)

    try:
        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        raw_transcript = RawTranscript.model_validate(raw_data)
    except Exception as e:
        msg = f"Failed to parse raw.json for {slug}: {e}"
        logger.error(msg)
        set_status(conn, video_id, Status.FAILED, failed_stage="correct", error=msg, stage="correct")
        raise StageError("correct", msg) from e

    if not dry_run:
        set_status(conn, video_id, Status.CORRECTING, stage="correct")

    # Step 2: Build CorrectedSegment list
    segments: list[CorrectedSegment] = [
        CorrectedSegment(**seg.model_dump(), raw_text=seg.text)
        for seg in raw_transcript.segments
    ]

    # Load glossary and rules
    glossary_terms: list[str] = []
    glossary_path = Path("glossary/aws.yml")
    glossary_version = 1
    if glossary_path.exists():
        try:
            g_file = load_glossary(glossary_path)
            glossary_terms = g_file.terms
            glossary_version = g_file.version
        except Exception as e:
            logger.warning("Failed to load glossary/aws.yml: %s", e)

    corrections_rules: list[Any] = []
    corrections_path = Path("glossary/corrections.yml")
    corrections_version = 1
    if corrections_path.exists():
        try:
            c_file = load_corrections(corrections_path)
            corrections_rules = c_file.corrections
            corrections_version = c_file.version
        except Exception as e:
            logger.warning("Failed to load glossary/corrections.yml: %s", e)

    applied_case_count = 0
    applied_rule_count = 0

    # Step 3: Deterministic pass per segment
    n_segs = len(segments)
    for i, seg in enumerate(segments):
        prev_text = segments[i - 1].text if i > 0 else ""
        next_text = segments[i + 1].text if i < n_segs - 1 else ""

        # 3a. Case normalization
        if glossary_terms:
            new_text, case_applied = apply_case_normalization(seg.text, glossary_terms)
            if case_applied:
                seg.text = new_text
                seg.corrected = True
                seg.correction = CorrectionInfo(
                    source="glossary_case",
                    reason=f"Case normalization: {', '.join(case_applied)}",
                )
                applied_case_count += 1

        # 3b. Auto rules
        if corrections_rules:
            new_text, rule_hits = apply_auto_rules(prev_text, seg.text, next_text, corrections_rules)
            if rule_hits:
                seg.text = new_text
                seg.corrected = True
                seg.correction = CorrectionInfo(
                    source="rule",
                    reason="; ".join(h.reason for h in rule_hits),
                )
                applied_rule_count += 1

    # Step 4: Flag pass on post-deterministic text
    flagged_segments: list[CorrectedSegment] = []
    for i, seg in enumerate(segments):
        prev_text = segments[i - 1].text if i > 0 else ""
        next_text = segments[i + 1].text if i < n_segs - 1 else ""
        flags = flag_segment(seg, prev_text, next_text, glossary_terms, corrections_rules, settings)
        seg.flags = flags
        if flags:
            flagged_segments.append(seg)

    suspicious_count = len(flagged_segments)

    # Step 5 & 6: LLM Correction pass
    tokens_in = 0
    tokens_out = 0
    applied_llm_count = 0
    rejected_count = 0
    sent_to_llm_count = 0
    llm_host: str | None = None
    created_client = False

    if settings.llm_enabled and not dry_run and flagged_segments:
        if client is None:
            settings.validate_llm()
            client = LlmClient(settings)
            created_client = True

        llm_host = client.host
        print(f"LLM: {len(flagged_segments)} segment(s) -> {llm_host} ({settings.llm_model})")

        # Determine cached vs uncached
        uncached_segs: list[CorrectedSegment] = []
        cached_results: dict[int, dict[str, Any]] = {}

        for seg in flagged_segments:
            prev_text = segments[seg.id - 1].text if seg.id > 0 else ""
            next_text = segments[seg.id + 1].text if seg.id < n_segs - 1 else ""
            flags_str = ",".join(sorted(seg.flags))
            hash_str = f"{settings.llm_model}:v1:{prev_text}:{seg.text}:{next_text}:{flags_str}"
            req_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()

            # Check llm_cache
            c_row = conn.execute(
                "SELECT response_json FROM llm_cache WHERE video_id = ? AND segment_id = ? AND request_hash = ?",
                (video_id, seg.id, req_hash),
            ).fetchone()

            if c_row:
                try:
                    cached_data = json.loads(c_row["response_json"])
                    cached_results[seg.id] = cached_data
                except Exception:
                    uncached_segs.append(seg)
            else:
                uncached_segs.append(seg)

        sent_to_llm_count = len(uncached_segs)

        # Batch uncached segments
        batch_size = max(settings.llm_batch_size, 1)
        batches: list[list[CorrectedSegment]] = [
            uncached_segs[i : i + batch_size]
            for i in range(0, len(uncached_segs), batch_size)
        ]

        llm_responses: dict[int, dict[str, Any]] = dict(cached_results)

        def process_batch(batch: list[CorrectedSegment]) -> tuple[dict[int, dict[str, Any]], int, int]:
            items_payload = []
            for b_seg in batch:
                p_text = segments[b_seg.id - 1].text if b_seg.id > 0 else ""
                n_text = segments[b_seg.id + 1].text if b_seg.id < n_segs - 1 else ""
                items_payload.append({
                    "id": b_seg.id,
                    "prev": p_text,
                    "text": b_seg.text,
                    "next": n_text,
                    "flags": b_seg.flags,
                })

            user_msg = build_user_message(glossary_terms, items_payload)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            max_tokens = 220 * len(batch)
            res_dict, usage = client.complete(messages, CORRECTION_SCHEMA, max_tokens)

            batch_res: dict[int, dict[str, Any]] = {}
            for res_item in res_dict.get("results", []):
                if "id" in res_item:
                    batch_res[int(res_item["id"])] = res_item

            return batch_res, usage.prompt_tokens, usage.completion_tokens

        if batches:
            try:
                with ThreadPoolExecutor(max_workers=settings.llm_workers) as executor:
                    future_to_batch = {executor.submit(process_batch, b): b for b in batches}
                    for future in as_completed(future_to_batch):
                        batch_res, p_tokens, c_tokens = future.result()
                        tokens_in += p_tokens
                        tokens_out += c_tokens
                        llm_responses.update(batch_res)

                        # Write to llm_cache atomically
                        now_iso = utc_now()
                        for seg_id, res_item in batch_res.items():
                            s_seg = next((s for s in flagged_segments if s.id == seg_id), None)
                            if s_seg:
                                p_text = segments[s_seg.id - 1].text if s_seg.id > 0 else ""
                                n_text = segments[s_seg.id + 1].text if s_seg.id < n_segs - 1 else ""
                                flags_str = ",".join(sorted(s_seg.flags))
                                hash_str = f"{settings.llm_model}:v1:{p_text}:{s_seg.text}:{n_text}:{flags_str}"
                                req_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()
                                with conn:
                                    conn.execute(
                                        """
                                        INSERT OR REPLACE INTO llm_cache (
                                            video_id, segment_id, request_hash, response_json, created_at
                                        ) VALUES (?, ?, ?, ?, ?)
                                        """,
                                        (video_id, seg_id, req_hash, json.dumps(res_item), now_iso),
                                    )
            except LlmUnavailable as e:
                logger.error("LLM unavailable during correction of %s: %s", slug, e)
                set_status(conn, video_id, Status.FAILED, failed_stage="correct", error=str(e), stage="correct")
                if created_client:
                    client.close()
                raise
            except Exception as e:
                logger.error("LLM correction failed for %s: %s", slug, e)
                set_status(conn, video_id, Status.FAILED, failed_stage="correct", error=str(e), stage="correct")
                if created_client:
                    client.close()
                raise

        if created_client:
            client.close()

        # Step 6b: Apply validation gate to each returned item
        for seg in flagged_segments:
            if seg.id not in llm_responses:
                # Segment was absent from LLM response
                seg.needs_review = True
                continue

            item = llm_responses[seg.id]
            changed = item.get("changed", False)

            if not changed:
                seg.needs_review = bool(item.get("needs_review", False))
                continue

            accepted, rejected_reason = validate_correction_item(seg.raw_text, item)
            if accepted:
                seg.text = str(item.get("corrected")).strip()
                seg.corrected = True
                seg.needs_review = bool(item.get("needs_review", False))
                seg.correction = CorrectionInfo(
                    source="llm",
                    reason=item.get("reason", "LLM correction"),
                    confidence=item.get("confidence"),
                    model=settings.llm_model,
                )
                applied_llm_count += 1
            else:
                seg.needs_review = True
                seg.correction = CorrectionInfo(
                    source="llm",
                    reason=item.get("reason", ""),
                    confidence=item.get("confidence"),
                    model=settings.llm_model,
                    rejected_reason=rejected_reason,
                )
                rejected_count += 1

    # Step 7: Segments still carrying flags when LLM not run -> needs_review = True
    if not settings.llm_enabled or dry_run:
        for seg in flagged_segments:
            seg.needs_review = True

    needs_review_count = sum(1 for s in segments if s.needs_review)

    # Step 8: Build metadata and output artifacts
    if settings.llm_base_url:
        parsed_url = urlparse(settings.llm_base_url)
        llm_endpoint_host = parsed_url.netloc or parsed_url.path
    else:
        llm_endpoint_host = None

    stats_meta = {
        "flagged": suspicious_count,
        "sent_to_llm": sent_to_llm_count,
        "applied_llm": applied_llm_count,
        "applied_rule": applied_rule_count,
        "applied_case": applied_case_count,
        "rejected": rejected_count,
        "needs_review": needs_review_count,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }

    correction_meta = CorrectionMeta(
        enabled=settings.llm_enabled,
        llm_model=settings.llm_model,
        llm_endpoint_host=llm_endpoint_host,
        glossary_version=glossary_version,
        corrections_version=corrections_version,
        created_at=utc_now(),
        stats=stats_meta,
    )

    corrected_transcript = CorrectedTranscript(
        video=raw_transcript.video,
        transcription=raw_transcript.transcription,
        correction=correction_meta,
        segments=segments,
    )

    # Build structured diff for corrections.json
    diff_items: list[CorrectionDiffItem] = []
    for seg in segments:
        if seg.corrected or seg.needs_review or seg.flags:
            source = seg.correction.source if seg.correction else "unmodified"
            reason = seg.correction.reason if seg.correction else None
            conf = seg.correction.confidence if seg.correction else None
            rej = seg.correction.rejected_reason if seg.correction else None
            diff_items.append(
                CorrectionDiffItem(
                    id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    raw=seg.raw_text,
                    corrected=seg.text,
                    source=source,
                    reason=reason,
                    confidence=conf,
                    accepted=seg.corrected,
                    rejected_reason=rej,
                    flags=seg.flags,
                )
            )

    corrections_file = CorrectionsFile(
        video=slug,
        stats=stats_meta,
        changes=diff_items,
    )

    # Step 9: Save artifacts unless dry_run
    if dry_run:
        print(
            f"[dry-run] {slug}: {suspicious_count} flagged segments, "
            f"target endpoint: {llm_endpoint_host or 'disabled'}"
        )
        return corrected_transcript

    atomic_write_json(corrected_json_path, corrected_transcript.model_dump())
    atomic_write_json(corrections_json_path, corrections_file.model_dump())

    # Step 10: Update database row
    # Status stays TRANSCRIBED until export stage sets COMPLETED or NEEDS_REVIEW
    set_status(
        conn,
        video_id,
        Status.TRANSCRIBED,
        stage="correct",
        corrected_count=applied_case_count + applied_rule_count + applied_llm_count,
        needs_review_count=needs_review_count,
        suspicious_count=suspicious_count,
        llm_tokens_in=tokens_in,
        llm_tokens_out=tokens_out,
    )

    return corrected_transcript
