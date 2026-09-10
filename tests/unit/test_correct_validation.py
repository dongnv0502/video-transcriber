"""Unit tests for correct.py: validation gate rules (accept/reject matrix)."""

import pytest

from transcriber.stages.correct import validate_correction_item


def test_gate_changed_false_keeps_raw_text() -> None:
    raw = "chúng ta cấu hình máy chủ"
    accepted, rej = validate_correction_item(raw, {"changed": False, "needs_review": True})
    assert accepted is False
    assert rej is None


def test_gate_empty_correction_rejected() -> None:
    raw = "chúng ta cấu hình máy chủ"
    accepted, rej = validate_correction_item(raw, {"changed": True, "corrected": "", "confidence": 0.95})
    assert accepted is False
    assert rej == "empty_correction"


def test_gate_identical_correction_treated_as_unchanged() -> None:
    raw = "chúng ta cấu hình máy chủ"
    accepted, rej = validate_correction_item(raw, {"changed": True, "corrected": raw, "confidence": 0.95})
    assert accepted is False
    assert rej == "identical_to_raw"


def test_gate_confidence_below_threshold_rejected() -> None:
    raw = "chúng ta cấu hình máy chủ"
    accepted, rej = validate_correction_item(
        raw,
        {"changed": True, "corrected": "chúng ta cấu hình máy chủ AWS", "confidence": 0.84},
    )
    assert accepted is False
    assert rej == "confidence_below_0.85"


def test_gate_fuzz_ratio_below_60_rejected() -> None:
    raw = "chúng ta cấu hình máy chủ"
    accepted, rej = validate_correction_item(
        raw,
        {"changed": True, "corrected": "hoàn toàn không liên quan gì", "confidence": 0.95},
    )
    assert accepted is False
    assert rej == "fuzz_ratio_below_60"


def test_gate_length_ratio_outside_bounds_rejected() -> None:
    raw = "chúng ta cấu hình máy chủ một hai ba bốn năm sáu bảy tám chín"
    short = "chúng ta cấu hình máy chủ một hai ba"  # len ratio < 0.7, fuzz >= 60
    accepted, rej = validate_correction_item(
        raw,
        {"changed": True, "corrected": short, "confidence": 0.95},
    )
    assert accepted is False
    assert rej == "length_ratio_outside_0.7_1.4"


def test_gate_numbers_modified_rejected() -> None:
    raw = "chúng ta khởi tạo 3 máy chủ"
    accepted, rej = validate_correction_item(
        raw,
        {"changed": True, "corrected": "chúng ta khởi tạo 5 máy chủ", "confidence": 0.95},
    )
    assert accepted is False
    assert rej == "numbers_modified"


def test_gate_word_count_diff_exceeds_3_rejected() -> None:
    raw = "chúng ta có một hệ thống máy chủ rất tốt ở đây"
    # 5 extra words added (paraphrase/expansion)
    expanded = "chúng ta có một hệ thống máy chủ rất tốt ở đây a b c d e"
    accepted, rej = validate_correction_item(
        raw,
        {"changed": True, "corrected": expanded, "confidence": 0.95},
    )
    assert accepted is False
    assert rej == "word_count_diff_exceeds_3"


def test_gate_valid_correction_accepted() -> None:
    raw = "chúng ta cấu hình bảo mật cho hệ thống"
    corrected = "chúng ta cấu hình bảo mật cho hệ thống AWS"
    accepted, rej = validate_correction_item(
        raw,
        {"changed": True, "corrected": corrected, "confidence": 0.95, "needs_review": False},
    )
    assert accepted is True
    assert rej is None
