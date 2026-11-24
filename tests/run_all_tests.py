"""
MarketPulse AI - Manual Test Runner
=======================================
Runs all test_*.py files' test_ functions without requiring pytest to be
installed (used in the offline development sandbox for this project; if
pytest is available, prefer `pytest` directly for richer output).

Run:
    python tests/run_all_tests.py
"""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_MODULES = [
    "tests.test_data_pipeline",
    "tests.test_analytics",
    "tests.test_models",
    "tests.test_mmm_pipeline",
    "tests.test_ai_analyst",
    "tests.test_api",
]


def run():
    total = 0
    passed = 0
    failed = []

    for mod_name in TEST_MODULES:
        print(f"\n=== {mod_name} ===")
        try:
            module = importlib.import_module(mod_name)
        except Exception as e:
            print(f"  [IMPORT ERROR] {e}")
            failed.append((mod_name, "import", str(e)))
            continue

        for name in dir(module):
            if name.startswith("test_"):
                fn = getattr(module, name)
                total += 1
                try:
                    fn()
                    print(f"  PASS: {name}")
                    passed += 1
                except Exception as e:
                    print(f"  FAIL: {name} -> {e}")
                    traceback.print_exc(limit=1)
                    failed.append((mod_name, name, str(e)))

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed")
    if failed:
        print("\nFailures:")
        for mod, name, err in failed:
            print(f"  - {mod}.{name}: {err}")
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    run()
