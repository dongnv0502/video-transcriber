"""Prompt templates and JSON schemas for LLM transcript correction."""

import json
from typing import Any

SYSTEM_PROMPT = (
    "Bạn là bộ hiệu đính ASR cho bài giảng AWS tiếng Việt. "
    "Chỉ được sửa lỗi nhận dạng giọng nói và chuẩn hóa thuật ngữ AWS (tên service, acronym, lệnh CLI). "
    "TUYỆT ĐỐI KHÔNG: tóm tắt, diễn giải lại, rút gọn, dịch, thêm thông tin mới, thay đổi con số. "
    "Nếu không chắc chắn: trả về changed=false và needs_review=true. Giữ nguyên văn phong và dấu câu của người giảng. "
    "LƯU Ý QUAN TRỌNG: Các trường 'prev' và 'next' trong mỗi item chỉ là ngữ cảnh tham khảo, "
    "TUYỆT ĐỐI KHÔNG được đưa nội dung của 'prev' hoặc 'next' vào kết quả 'corrected'. "
    "Chỉ sửa duy nhất phần câu trong trường 'text'."
)

CORRECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["results"],
    "properties": {
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": False,
          "required": ["id", "changed", "corrected", "reason", "confidence", "needs_review"],
          "properties": {
            "id": {"type": "integer"},
            "changed": {"type": "boolean"},
            "corrected": {"type": ["string", "null"]},
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
            "needs_review": {"type": "boolean"},
          },
        },
      }
    },
}


def build_user_message(glossary_terms: list[str], items: list[dict[str, Any]]) -> str:
    """Construct compact JSON user prompt containing glossary and flagged segments."""
    # Cap glossary terms to at most 40 terms to optimize prompt tokens
    selected_terms = glossary_terms[:40] if len(glossary_terms) > 40 else glossary_terms
    payload = {
        "glossary": selected_terms,
        "items": items,
    }
    return json.dumps(payload, ensure_ascii=False)
