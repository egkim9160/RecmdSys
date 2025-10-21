#!/usr/bin/env python3
"""LLM and OpenAI client utilities"""
import os
import json
import logging
from typing import Optional, List

# Setup logger
logger = logging.getLogger(__name__)


def get_openai_client():
    """Create an OpenAI client if keys exist. Returns None if not configured."""
    try:
        from openai import OpenAI
    except Exception:
        logger.warning("[LLM] OpenAI library not available")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("[LLM] OPENAI_API_KEY not set")
        return None

    base_url = os.getenv("OPENAI_BASE_URL")

    try:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"[LLM] OpenAI client initialized with base_url={base_url}")
        else:
            client = OpenAI(api_key=api_key)
            logger.info("[LLM] OpenAI client initialized with default URL")
        return client
    except Exception as e:
        logger.error(f"[LLM] Failed to initialize OpenAI client: {e}")
        return None


def get_embedding_client():
    """Create an OpenAI embedding client"""
    try:
        from openai import OpenAI
    except Exception:
        logger.warning("[EMBED] OpenAI library not available")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("[EMBED] OPENAI_API_KEY not set")
        return None
    base_url = os.getenv("OPENAI_BASE_URL") or "https://llm-api.medigate.net/embedding/v1"
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=240)
        logger.info(f"[EMBED] Embedding client initialized with base_url={base_url}")
        return client
    except Exception as e:
        logger.error(f"[EMBED] Failed to initialize embedding client: {e}")
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
    except Exception:
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
        logger.warning("[LLM] clean_address_with_llm called but client is None")
        return None

    model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    system_prompt = (
        "너는 한국 주소 정제기야. 입력 주소에서 건물명/아파트명/호수/층/동/상세호 등을 제거하고, "
        "도로명+건물번호 또는 지번까지만 남겨. 애매한 행정구역명은 공식명칭으로 바꿔줘. \n"
        "출력은 오직 정제된 주소 한 줄만 내보내. 불필요한 말/따옴표/설명은 금지."
    )
    user_prompt = f"원본 주소: {raw_address}\n정제된 주소:"

    logger.info(f"[LLM] clean_address_with_llm request: raw_address='{raw_address}'")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content.strip() if resp and resp.choices else None
        if not text:
            logger.warning(f"[LLM] clean_address_with_llm: empty response for '{raw_address}'")
            return None
        # Heuristic: remove wrapping quotes if present
        text = text.strip().strip('"').strip("'")
        # Very short outputs are suspicious
        if len(text) < 4:
            logger.warning(f"[LLM] clean_address_with_llm: response too short ('{text}') for '{raw_address}'")
            return None
        logger.info(f"[LLM] clean_address_with_llm success: '{raw_address}' -> '{text}'")
        return text
    except Exception as e:
        logger.error(f"[LLM] clean_address_with_llm error for '{raw_address}': {e}")
        return None
