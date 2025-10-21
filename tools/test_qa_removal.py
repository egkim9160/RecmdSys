#!/usr/bin/env python3
"""Test Q&A section removal"""

import re


def _remove_qa_section(text: str) -> str:
    """Remove Q&A section that starts with 'Q.' from the text"""
    # Find the first occurrence of Q. (with possible whitespace/newlines before)
    # and remove everything from that point onwards
    pattern = r'[\s\n]*Q\.\s.*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


# Test case
test_text = """병원소개
안녕하세요 선생님 경기도에 위치한 종합병원에서 신경과 선생님을 초빙중입니다.
서울 접근성이 좋아 장기 근속 하는 분들이 많으며, 시스템이 잘 갖춰져 있어 적극 추천 드립니다.
신경과는 총 2인으로 운영됩니다.

근무조건
급여 Net 1.7~1.8 (경력에 따라 협의 가능합니다)
업무내용: 외래 및 입원환자 관리
토요일 격주 근무, 토요일 근무 하는 주는 평일 하루 휴무
오프는 평일 1일 반차 사용 가능합니다.

Q. 구직시에 비용이 들어가나요?
아니오, 메디게이트 회원분들은 무료로 이용 가능합니다.

Q. H+LINK(에이치링크)서비스를 이용하는 것과, 직접 지원하는 것은  무엇이 다른가요?
공고에는 없고, 에이치링크에만 있는 병원도 있으며, 병원에 직접 문의하지 않아도 근무 조건을 정확하게 알 수 있습니다.
또한 연봉 협상 등 병원과 직접 하기 어려운 이야기들을 에이치링크 컨설턴트가 대신 진행해 드립니다.
나에게 맞는 병원인지 아닌지, 편하게 문의해 보세요!"""

print("="*80)
print("ORIGINAL TEXT:")
print("="*80)
print(test_text)
print()

cleaned = _remove_qa_section(test_text)

print("="*80)
print("CLEANED TEXT (Q&A removed):")
print("="*80)
print(cleaned)
print()

print("="*80)
print("SUMMARY:")
print("="*80)
print(f"Original length: {len(test_text)} chars")
print(f"Cleaned length: {len(cleaned)} chars")
print(f"Removed: {len(test_text) - len(cleaned)} chars")
print(f"Removed: {((len(test_text) - len(cleaned)) / len(test_text) * 100):.1f}%")
