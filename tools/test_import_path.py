#!/usr/bin/env python3
"""Test if module imports work correctly"""

import sys
from pathlib import Path

print("="*80)
print("IMPORT PATH TEST")
print("="*80)

# Check current working directory
print(f"\n[CWD] {Path.cwd()}")

# Check script location
script_path = Path(__file__).resolve()
print(f"\n[SCRIPT] {script_path}")
print(f"[SCRIPT DIR] {script_path.parent}")
print(f"[PROJECT ROOT] {script_path.parent.parent}")

# Check sys.path
print(f"\n[SYS.PATH]")
for i, p in enumerate(sys.path[:10], 1):
    print(f"  {i}. {p}")

# Add project root to path (like in hlink_infer_pipeline.py)
project_root = script_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"\n[ADDED] {project_root} to sys.path")

# Try importing module
print("\n" + "="*80)
print("TESTING IMPORTS")
print("="*80)

try:
    from module.db_utils import get_connection
    print("[✓] from module.db_utils import get_connection")
except ImportError as e:
    print(f"[✗] from module.db_utils import get_connection")
    print(f"    Error: {e}")

try:
    from module.naver_geo import geocode_naver
    print("[✓] from module.naver_geo import geocode_naver")
except ImportError as e:
    print(f"[✗] from module.naver_geo import geocode_naver")
    print(f"    Error: {e}")

try:
    from module.llm_utils import get_openai_client
    print("[✓] from module.llm_utils import get_openai_client")
except ImportError as e:
    print(f"[✗] from module.llm_utils import get_openai_client")
    print(f"    Error: {e}")

try:
    from module.html_utils import clean_html_and_get_urls
    print("[✓] from module.html_utils import clean_html_and_get_urls")
except ImportError as e:
    print(f"[✗] from module.html_utils import clean_html_and_get_urls")
    print(f"    Error: {e}")

print("\n" + "="*80)
print("DIRECTORY STRUCTURE")
print("="*80)

# Check if module directory exists
module_dir = project_root / "module"
if module_dir.exists():
    print(f"\n[✓] Module directory exists: {module_dir}")
    print(f"\n[MODULE FILES]")
    for py_file in sorted(module_dir.glob("*.py")):
        print(f"  - {py_file.name}")
else:
    print(f"\n[✗] Module directory NOT found: {module_dir}")

# Check if tools/module exists (wrong location)
tools_module_dir = project_root / "tools" / "module"
if tools_module_dir.exists():
    print(f"\n[⚠️] WARNING: tools/module exists (wrong location): {tools_module_dir}")
else:
    print(f"\n[✓] tools/module does NOT exist (correct)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Project structure is correct if:")
print(f"  1. RecmdSys/module/ exists (not RecmdSys/tools/module/)")
print(f"  2. All 'from module.xxx' imports work")
print(f"  3. sys.path includes project root")
