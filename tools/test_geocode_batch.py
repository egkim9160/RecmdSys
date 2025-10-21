#!/usr/bin/env python3
"""Test batch geocoding with debug logging"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load env
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
except Exception:
    pass

# Import after path setup
from export_candidates_from_opensearch import geocode
from concurrent.futures import ThreadPoolExecutor, as_completed

# Test addresses (sample from real data)
test_addresses = [
    "서울 강남구 테헤란로 152",
    "경기 성남시 분당구 서현동 255",
    "서울 서초구 반포대로 222",
    "경기 용인시 기흥구 보정동 1234",
    "서울 송파구 올림픽로 300",
]

print("="*80)
print("TESTING BATCH GEOCODING WITH DEBUG")
print("="*80)

print("\n[TEST 1] Direct calls (sequential)")
print("-"*80)
for addr in test_addresses[:3]:
    print(f"\nTesting: {addr}")
    lat, lon = geocode(addr)
    print(f"Result: lat={lat}, lon={lon}")

print("\n" + "="*80)
print("[TEST 2] ThreadPoolExecutor (parallel)")
print("-"*80)

def task(addr):
    return addr, geocode(addr)

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(task, addr) for addr in test_addresses]
    for i, future in enumerate(as_completed(futures), 1):
        addr, (lat, lon) = future.result()
        print(f"\n[{i}/{len(test_addresses)}] Address: {addr}")
        print(f"           Result: lat={lat}, lon={lon}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
