#!/usr/bin/env python3
"""LLM and OpenAI client utilities - v2 with Gemini embedding support"""
import os
import json
import re
from typing import Optional, List, Tuple


# ============================================================================
# Gemini Embedding 설정
# ============================================================================
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_EMBEDDING_DIMENSION = 3072
GEMINI_SUMMARY_MODEL = "gemini-2.5-flash"  # 요약용 모델
GEMINI_TOKEN_LIMIT = 2000  # 토큰 초과 시 요약 수행

# Gemini 클라이언트 (싱글톤)
_gemini_client = None


def get_openai_client():
    """Create an OpenAI client if keys exist. Returns None if not configured."""
    try:
        from openai import OpenAI
    except Exception:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL")

    try:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


def get_embedding_client():
    """Create an OpenAI embedding client"""
    try:
        from openai import OpenAI
    except Exception:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    # 고정 엔드포인트 사용 (요청 사항)
    base_url = "https://llm-api.medigate.net/embedding/v1"
    try:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=240)
    except Exception:
        return None


def batch_embed_texts(client, texts: List[str], model: str = "text-embedding-3-large", batch_size: int = 64) -> List:
    """Batch embed texts using OpenAI API"""
    embeddings = [None] * len(texts)
    if client is None:
        return embeddings
    try:
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            sub = texts[start:end]
            resp = client.embeddings.create(input=sub, model=model)
            vecs = [item.embedding for item in resp.data]
            for i, v in enumerate(vecs):
                embeddings[start + i] = v
        return embeddings
    except Exception as embed_err:
        try:
            print(f"[EMBED][WARN] embedding request failed: {embed_err}")
        except Exception:
            pass
        return embeddings


def clean_address_with_llm(raw_address: str, client) -> Optional[str]:
    """
    Use LLM to normalize a messy Korean address string:
      - Remove building/apartment names, floors, room numbers
      - Keep up to lot number or road-name + building number
      - Normalize ambiguous admin divisions to official names
      - Return ONLY the cleaned address as plain text
    Returns None if client is unavailable or on failure.
    """
    if client is None:
        return None

    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    system_prompt = (
        "너는 한국 주소 정제기야. 입력 주소에서 건물명/아파트명/호수/층/동/상세호 등을 제거하고, "
        "도로명+건물번호 또는 지번까지만 남겨. 애매한 행정구역명은 공식명칭으로 바꿔줘. \n"
        "출력은 오직 정제된 주소 한 줄만 내보내. 불필요한 말/따옴표/설명은 금지."
    )
    user_prompt = f"원본 주소: {raw_address}\n정제된 주소:"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        text = resp.choices[0].message.content.strip() if resp and resp.choices else None
        if not text:
            return None
        # Heuristic: remove wrapping quotes if present
        text = text.strip().strip('"').strip("'")
        # Very short outputs are suspicious
        if len(text) < 4:
            return None
        return text
    except Exception:
        return None


# ============================================================================
# Gemini Embedding Functions (gemini-embedding-001, 3072d, SEMANTIC_SIMILARITY)
# ============================================================================

def get_gemini_client():
    """Gemini 클라이언트 초기화 (싱글톤)"""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    try:
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[WARN] GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.", flush=True)
            return None
        _gemini_client = genai.Client(api_key=api_key)
        return _gemini_client
    except Exception as e:
        print(f"[WARN] Gemini 클라이언트 초기화 실패: {e}", flush=True)
        return None


def count_gemini_tokens(text: str) -> int:
    """gemini-embedding-001 기준 토큰 수 계산"""
    client = get_gemini_client()
    if client is None:
        return len(text) // 4  # 대략적인 추정
    try:
        result = client.models.count_tokens(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text
        )
        return result.total_tokens
    except Exception as e:
        print(f"[WARN] 토큰 계산 실패: {e}", flush=True)
        return len(text) // 4


def clean_html_simple(text: str) -> str:
    """HTML 태그 및 특수문자 정리 (간단 버전)"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def summarize_text_for_embedding(text: str, retry: bool = True) -> Tuple[str, bool]:
    """
    긴 텍스트 요약 (Gemini 사용)

    Args:
        text: 요약할 텍스트
        retry: 실패 시 재시도 여부

    Returns:
        (요약된 텍스트, 성공 여부)
    """
    client = get_gemini_client()
    if client is None:
        return text, False

    system_prompt = """당신은 채용공고/이력서 요약 전문가입니다.
주어진 텍스트를 핵심 정보 중심으로 간결하게 요약하세요.

요약 시 포함해야 할 정보:
1. 직종/전문과
2. 위치/지역
3. 급여/처우 조건
4. 경력/자격 요건
5. 기타 특이사항

응답 형식:
- 문장 형태로 자연스럽게 작성
- 500자 이내로 간결하게
- 불필요한 서식이나 기호 제외"""

    try:
        from google.genai import types
        response = client.models.generate_content(
            model=GEMINI_SUMMARY_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=f"{system_prompt}\n\n다음 텍스트를 요약해주세요:\n\n{text}")]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            )
        )

        summary = response.text.strip() if response.text else None
        if summary and len(summary) > 10:
            return summary, True
        return text, False

    except Exception as e:
        print(f"[WARN] 요약 실패: {e}", flush=True)
        if retry:
            return summarize_text_for_embedding(text, retry=False)
        return text, False


def embed_text_gemini(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> Optional[List[float]]:
    """
    Gemini 임베딩 생성 (gemini-embedding-001, 3072 차원)

    Args:
        text: 임베딩할 텍스트
        task_type: 태스크 타입 (SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT 등)

    Returns:
        임베딩 벡터 또는 None
    """
    client = get_gemini_client()
    if client is None:
        return None

    try:
        from google.genai import types
        result = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=GEMINI_EMBEDDING_DIMENSION
            )
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"[ERROR] Gemini 임베딩 실패: {e}", flush=True)
        return None


def process_text_for_gemini_embedding(text: str, task_type: str = "SEMANTIC_SIMILARITY") -> Tuple[Optional[List[float]], bool, int]:
    """
    텍스트를 Gemini embedding으로 변환 (토큰 초과 시 자동 요약)

    Args:
        text: 원본 텍스트
        task_type: 임베딩 태스크 타입

    Returns:
        (임베딩 벡터, 요약 수행 여부, 토큰 수)
    """
    if not text or not text.strip():
        return None, False, 0

    # HTML 정리
    cleaned_text = clean_html_simple(text)

    # 토큰 계산
    token_count = count_gemini_tokens(cleaned_text)
    summarized = False

    # 토큰 > 2000이면 요약
    if token_count > GEMINI_TOKEN_LIMIT:
        cleaned_text, success = summarize_text_for_embedding(cleaned_text)
        summarized = True
        if not success:
            print(f"[WARN] 요약 실패, 원본 텍스트 앞부분만 사용 (토큰: {token_count})", flush=True)
            # 요약 실패 시 앞부분만 잘라서 사용
            cleaned_text = cleaned_text[:8000]  # 대략 2000 토큰 정도

    # 임베딩 생성
    embedding = embed_text_gemini(cleaned_text, task_type=task_type)

    return embedding, summarized, token_count


def batch_embed_texts_gemini(
    texts: List[str],
    task_type: str = "SEMANTIC_SIMILARITY",
    verbose: bool = False,
    log_interval: int = 100,
    max_workers: int = 10
) -> List[Optional[List[float]]]:
    """
    Gemini를 사용한 배치 임베딩 (병렬 처리)

    Args:
        texts: 텍스트 리스트
        task_type: 임베딩 태스크 타입
        verbose: 진행 로그 출력 여부
        log_interval: 로그 출력 간격
        max_workers: 병렬 처리 워커 수 (기본 10)

    Returns:
        임베딩 벡터 리스트
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    import time

    embeddings = [None] * len(texts)
    summarized_count = 0
    success_count = 0
    lock = Lock()
    start_time = time.time()

    if verbose:
        print(f"[GEMINI_EMBED] 시작: {len(texts)}건 처리 예정 (workers={max_workers})...", flush=True)

    def process_single(idx: int, text: str) -> Tuple[int, Optional[List[float]], bool]:
        """단일 텍스트 임베딩 처리"""
        if not text or not text.strip():
            return idx, None, False
        vec, was_summarized, _ = process_text_for_gemini_embedding(text, task_type=task_type)
        return idx, vec, was_summarized

    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_idx = {
            executor.submit(process_single, i, text): i
            for i, text in enumerate(texts)
        }

        # 완료되는 순서대로 처리
        for future in as_completed(future_to_idx):
            try:
                idx, vec, was_summarized = future.result()
                embeddings[idx] = vec

                with lock:
                    if vec is not None:
                        success_count += 1
                    if was_summarized:
                        summarized_count += 1
                    done_count += 1

                    if verbose and (done_count % log_interval == 0 or done_count == len(texts)):
                        elapsed = time.time() - start_time
                        rate = done_count / elapsed if elapsed > 0 else 0
                        print(f"[GEMINI_EMBED] Progress: {done_count}/{len(texts)}, Success: {success_count}, Summarized: {summarized_count}, {rate:.1f}건/초", flush=True)

            except Exception as e:
                with lock:
                    done_count += 1
                if verbose:
                    print(f"[GEMINI_EMBED] Error at idx {future_to_idx[future]}: {e}", flush=True)

    if verbose:
        elapsed = time.time() - start_time
        print(f"[GEMINI_EMBED] Complete: {len(texts)} total, {success_count} success, {summarized_count} summarized ({elapsed:.1f}초)", flush=True)

    return embeddings
