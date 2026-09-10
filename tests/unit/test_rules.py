"""Unit tests for rules.py: case normalization, context gating, and candidate detection."""

from transcriber.stages.rules import (
    CorrectionRule,
    apply_auto_rules,
    apply_case_normalization,
    candidate_hits,
)


def test_case_normalization_only_changes_case() -> None:
    terms = ["IAM Role", "EC2", "S3", "VPC", "CloudFormation"]
    text = "sử dụng iam role để cho phép ec2 truy cập vào s3 và vpc"
    res, applied = apply_case_normalization(text, terms)
    assert res == "sử dụng IAM Role để cho phép EC2 truy cập vào S3 và VPC"
    assert "IAM Role" in applied
    assert "EC2" in applied
    assert "S3" in applied
    assert "VPC" in applied

    # Does not change already canonical terms
    canonical = "IAM Role EC2 S3"
    res2, applied2 = apply_case_normalization(canonical, terms)
    assert res2 == canonical
    assert len(applied2) == 0

    # Does not change words that only partially match
    unrelated = "s300 ec2000"
    res3, applied3 = apply_case_normalization(unrelated, terms)
    assert res3 == unrelated
    assert len(applied3) == 0


def test_requires_context_gating() -> None:
    rule = CorrectionRule(
        **{
            "from": "i am",
            "to": "IAM",
            "match": "literal",
            "auto": True,
            "requires_context": ["aws", "role", "policy"],
        }
    )

    # Context matched ('aws' in window)
    res_match, hits = apply_auto_rules(
        prev="trong bài giảng aws",
        text="chúng ta cấu hình i am user",
        next="",
        corrections=[rule],
    )
    assert "IAM user" in res_match
    assert len(hits) == 1

    # Context NOT matched
    res_no_match, hits_no = apply_auto_rules(
        prev="hôm nay trời đẹp",
        text="chúng ta cấu hình i am user",
        next="rất vui",
        corrections=[rule],
    )
    assert "i am user" in res_no_match
    assert len(hits_no) == 0


def test_auto_false_never_applies_automatically() -> None:
    manual_rule = CorrectionRule(
        **{
            "from": "ét ba",
            "to": "S3",
            "match": "literal",
            "auto": False,
            "requires_context": ["bucket", "storage"],
        }
    )
    # Even if context is satisfied, auto=False must NOT apply in apply_auto_rules
    res, hits = apply_auto_rules(
        prev="lưu file vào bucket",
        text="của ét ba",
        next="",
        corrections=[manual_rule],
    )
    assert res == "của ét ba"
    assert len(hits) == 0

    # But candidate_hits MUST detect it as a candidate
    candidates = candidate_hits("lưu file vào bucket", "của ét ba", "", [manual_rule])
    assert len(candidates) == 1
    assert "Candidate (manual)" in candidates[0]
