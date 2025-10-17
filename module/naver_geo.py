#!/usr/bin/env python3
# naver_geo.py
from __future__ import annotations
import os
import requests
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

NAVER_GEOCODE_URL = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
NAVER_DIRECTIONS_URL = "https://maps.apigw.ntruss.com/map-direction/v1/driving"
def _load_env_from_project_root() -> None:
    """Load .env from project root (where run_pipeline.py resides)"""
    try:
        project_root = Path(__file__).resolve().parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(str(env_path))
    except Exception:
        pass

_load_env_from_project_root()



def _get_naver_headers() -> Dict[str, str]:
    key_id = os.getenv("NAVER_API_CLIENT_ID") or os.getenv("NAVER_CLIENT_ID")
    key = os.getenv("NAVER_API_CLIENT_SECRET") or os.getenv("NAVER_CLIENT_SECRET")
    if not key_id or not key:
        raise EnvironmentError(
            "네이버 API 키가 설정되지 않았습니다. NAVER_API_KEY_ID / NAVER_API_KEY 환경변수를 설정하세요."
        )
    return {
        "x-ncp-apigw-api-key-id": key_id,
        "x-ncp-apigw-api-key": key,
    }


def geocode_naver(address: str) -> Tuple[float, float]:
    params = {"query": address}
    r = requests.get(NAVER_GEOCODE_URL, params=params, headers=_get_naver_headers(), timeout=10)
    r.raise_for_status()
    data = r.json()
    addresses = data.get("addresses", [])
    if not addresses:
        raise ValueError(f"지오코딩 실패: {data}")
    first = addresses[0]
    lon = float(first["x"])  # 경도
    lat = float(first["y"])  # 위도
    return (lat, lon)


def get_directions5_summary(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    *,
    option: str = "traoptimal",
    cartype: int = 1,
    waypoints: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    start_str = f"{start[1]},{start[0]}"  # lon,lat
    goal_str = f"{goal[1]},{goal[0]}"    # lon,lat

    params: Dict[str, str] = {
        "start": start_str,
        "goal": goal_str,
        "option": option,
        "cartype": str(cartype),
    }
    if waypoints:
        wp_str = "|".join(f"{lon},{lat}" for lat, lon in waypoints)
        params["waypoints"] = wp_str

    r = requests.get(NAVER_DIRECTIONS_URL, params=params, headers=_get_naver_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise ValueError(f"길찾기 실패: {data}")

    route = data.get("route", {})
    route_key = option if option in route else (next(iter(route.keys())) if route else None)
    if not route_key or not route.get(route_key):
        raise ValueError(f"길찾기 결과 없음: {data}")

    first_route = route[route_key][0]
    summary = first_route.get("summary", {})
    return {
        "distance_m": float(summary.get("distance", 0.0)),
        "duration_ms": float(summary.get("duration", 0.0)),
        "tollFare": float(summary.get("tollFare", 0.0)),
        "fuelPrice": float(summary.get("fuelPrice", 0.0)),
        "taxiFare": float(summary.get("taxiFare", 0.0)),
        "departureTime": summary.get("departureTime"),
    }


def calculate_haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    import math
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    return distance


def compute_distance_time_between_addresses(
    start_address: str,
    goal_address: str,
) -> Dict[str, Any]:
    start_coord = geocode_naver(start_address)
    goal_coord = geocode_naver(goal_address)
    haversine_km = calculate_haversine_distance(start_coord, goal_coord)
    driving = get_directions5_summary(start_coord, goal_coord, option="traoptimal", cartype=1)
    return {
        "start": {"lat": start_coord[0], "lon": start_coord[1]},
        "goal": {"lat": goal_coord[0], "lon": goal_coord[1]},
        "haversine_km": haversine_km,
        "driving_distance_km": driving["distance_m"] / 1000.0,
        "duration_min": driving["duration_ms"] / 60000.0,
        "tollFare": driving.get("tollFare", 0.0),
        "fuelPrice": driving.get("fuelPrice", 0.0),
        "taxiFare": driving.get("taxiFare", 0.0),
        "departureTime": driving.get("departureTime"),
    }


if __name__ == "__main__":
    s = "서울특별시 강남구"
    g = "부산광역시 부산진구"
    try:
        result = compute_distance_time_between_addresses(s, g)
        print("Start:", result["start"])  
        print("Goal:", result["goal"])  
        print("Haversine(km):", round(result["haversine_km"], 3))
        print("Driving Distance(km):", round(result["driving_distance_km"], 3))
        print("Duration(min):", round(result["duration_min"], 1))
        print("TollFare:", result["tollFare"])  
        print("FuelPrice:", result["fuelPrice"])  
    except Exception as e:
        print("에러:", e)


