#!/usr/bin/env python3
"""Geocoding utilities"""
import time
from typing import Optional, Tuple, Dict
from threading import Lock
from .naver_geo import geocode_naver


def try_geocode(
    address: Optional[str],
    cache: Dict[str, Tuple[float, float]],
    *,
    sleep_sec: float = 0.0,
    lock: Optional[Lock] = None
) -> Tuple[Optional[float], Optional[float], str]:
    """Attempt geocoding using NAVER. Use cache to reduce calls.
    Returns (lat, lon, status). status in {ok, empty, error}
    """
    if address is None:
        return (None, None, "empty")
    a = address.strip()
    if not a:
        return (None, None, "empty")

    if lock:
        with lock:
            if a in cache:
                lat, lon = cache[a]
                return (lat, lon, "ok")
    else:
        if a in cache:
            lat, lon = cache[a]
            return (lat, lon, "ok")

    try:
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        lat, lon = geocode_naver(a)
        if lock:
            with lock:
                cache[a] = (lat, lon)
        else:
            cache[a] = (lat, lon)
        return (lat, lon, "ok")
    except Exception:
        return (None, None, "error")
