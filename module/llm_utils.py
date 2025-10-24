#!/usr/bin/env python3
"""LLM and OpenAI client utilities"""
import os
import json
from typing import Optional, List


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
