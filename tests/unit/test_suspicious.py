"""Unit tests for suspicious.py: table-driven detection of all 8 anomaly flags."""

import pytest

from transcriber.models import RawSegment
from transcriber.stages.rules import CorrectionRule
from transcriber.stages.suspicious import (
    CORRECTION_CANDIDATE,
    FAST_TEXT,
    GLOSSARY_NEAR_MISS,
    HIGH_COMPRESSION,
    HIGH_NOSPEECH,
    LOW_LOGPROB,
    REPETITION,
    SHORT_SEGMENT,
    flag_segment,
)

TERMS = ["IAM", "EC2", "S3", "VPC", "CloudFormation", "Lambda", "Route 53"]
CANDIDATE_RULE = CorrectionRule(
    **{"from": "ét ba", "to": "S3", "match": "literal", "auto": False, "requires_context": []}
)


def test_clean_segment_no_flags() -> None:
    seg = RawSegment(
        id=0,
        start=0.0,
        end=3.0,
        text="chúng ta khởi tạo dịch vụ bình thường",
        avg_logprob=-0.20,
        no_speech_prob=0.01,
        compression_ratio=1.1,
    )
    flags = flag_segment(seg, "", "", TERMS, [])
    assert flags == []


@pytest.mark.parametrize(
    "seg_kwargs,expected_flag",
    [
        ({"avg_logprob": -0.75}, LOW_LOGPROB),
        ({"no_speech_prob": 0.65}, HIGH_NOSPEECH),
        ({"compression_ratio": 2.5}, HIGH_COMPRESSION),
        ({"text": "chúng ta chạy máy chủ máy chủ máy chủ máy chủ liên tục"}, REPETITION),
        ({"start": 0.0, "end": 1.0, "text": "đây là một câu nói nói nói rất là nhanh quá mức 25 ký tự trên giây"}, FAST_TEXT),
        ({"start": 0.0, "end": 0.2, "text": "đoạn này rất ngắn nhưng nhiều chữ"}, SHORT_SEGMENT),
        ({"text": "chúng ta dùng lamda để chạy hàm"}, GLOSSARY_NEAR_MISS),  # 'lamda' near-miss to 'Lambda'
    ],
)
def test_individual_anomaly_flags(seg_kwargs: dict, expected_flag: str) -> None:
    base = {
        "id": 1,
        "start": 0.0,
        "end": 3.0,
        "text": "nội dung hợp lệ",
        "avg_logprob": -0.20,
        "no_speech_prob": 0.01,
        "compression_ratio": 1.1,
    }
    base.update(seg_kwargs)
    seg = RawSegment(**base)
    flags = flag_segment(seg, "", "", TERMS, [])
    assert expected_flag in flags


def test_correction_candidate_flag() -> None:
    seg = RawSegment(
        id=2,
        start=0.0,
        end=2.0,
        text="lưu dữ liệu vào ét ba",
        avg_logprob=-0.2,
    )
    flags = flag_segment(seg, "", "", TERMS, [CANDIDATE_RULE])
    assert CORRECTION_CANDIDATE in flags
