#!/usr/bin/env python3
"""HTML cleaning and text extraction utilities"""
import re
import html
from typing import List, Tuple


def clean_html_and_get_urls(html_string: str) -> Tuple[str, List[str]]:
    """HTML에서 텍스트를 추출하고 태그를 제거해 문장 단위로 정리합니다. (이미지 URL 미사용)"""
    if not html_string:
        return "", []

    # 이미지 URL 추출 로직 제거 (요구사항 상 미사용)

    # 1) HTML 엔티티 먼저 디코딩 (인코딩된 태그도 실제 태그 형태로 변환)
    s = html.unescape(html_string)
    s = s.replace('\r\n', '\n').replace('\r', '\n')

    # 2) 주석/스크립트/스타일 컨텐츠 제거
    s = re.sub(r'<!--.*?-->', '', s, flags=re.S)
    s = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', s, flags=re.S | re.I)

    # 3) <br>와 블록 경계는 줄바꿈 보존, 그 외 모든 태그는 제거
    #    일단 <br> -> \n 로 치환
    s = re.sub(r'<\s*br\s*/?>', '\n', s, flags=re.I)
    #    나머지 모든 태그 제거 (깨진 조각 포함, 안전하게 전부 삭제)
    s = re.sub(r'<[^>]+>', ' ', s)

    # 4) NBSP/공백 정리
    s = s.replace('\u00A0', ' ')
    s = re.sub(r'&nbsp;+', ' ', s, flags=re.I)

    # 5) 라인화/노이즈 라인 제거/공백 축소
    lines = [ln.strip() for ln in s.split('\n')]
    def is_noise(line: str) -> bool:
        if not line:
            return True
        no_space = re.sub(r"\s+", "", line)
        if not no_space:
            return True
        if re.fullmatch(r"[-=~_*#.,·•·│┼┄┈┉━─–—·]+", no_space):
            return True
        return False
    lines = [ln for ln in lines if not is_noise(ln)]
    # 내부 다중 공백 축소
    lines = [re.sub(r'\s+', ' ', ln) for ln in lines]

    clean_text = '\n'.join(lines).strip()
    # 과도한 반복 문자 축소, 3개 이상 연속 빈 줄 축소
    clean_text = re.sub(r'([\-._*#=])\1{2,}', r'\1\1', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)

    return clean_text, []
