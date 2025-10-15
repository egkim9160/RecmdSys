#!/usr/bin/env python3
"""Common data processing utilities"""
import re
import math
from typing import Optional


def to_str_safe(value: Optional[str]) -> str:
    """Convert to string safely. Returns empty string for non-strings/NaN."""
    if isinstance(value, str):
        return value.strip()
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return (str(value).strip()) if value is not None else ""


def normalize_career_years(value: Optional[str]) -> int:
    """Map textual career years to integer years.
    Examples: '1~2년'->1, '2~3년'->2, '5~6년'->5, '10년 이상'->10, '1년 미만'->0, '경력 없음'->0
    Default fallback is 0 when unrecognized.
    """
    s = to_str_safe(value)
    if not s:
        return 0
    s = s.replace(" ", "")
    # Range like 1~2년, 2-3년
    m = re.match(r"^(\d+)[~\-∼](\d+)년$", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    # N년 이상
    m = re.match(r"^(\d+)년이상$", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    # N년 미만
    m = re.match(r"^(\d+)년미만$", s)
    if m:
        try:
            n = int(m.group(1))
            return max(0, n - 1)
        except Exception:
            return 0
    # 경력 없음
    if "경력없음" in s:
        return 0
    # 단일 숫자년
    m = re.match(r"^(\d+)년$", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def compute_specialty(major: Optional[str], detail: Optional[str]) -> Optional[str]:
    """Compute specialty from major and detail specialty"""
    major_s = to_str_safe(major)
    detail_s = to_str_safe(detail)

    if major_s == "내과":
        if detail_s and detail_s != "세부분과없음":
            return detail_s
        return "내과"
    return major_s or None
