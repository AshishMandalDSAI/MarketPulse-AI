"""Hardcode detection tests — Phase 0.

Static analysis guards that detect hardcoded column names, file paths,
and schema assumptions in the UPLOADED-DATASET code paths (data_engine,
generic KPIs, capability engine).

These tests do NOT ban hardcoded strings globally — the legacy NovaMart
modules (src/models/, src/forecasting/, src/optimization/) are allowed
to retain their hardcoded references for now. This test file specifically
targets the dataset-agnostic layer that must remain schema-free.

Targeted modules (must NOT contain hardcoded schema assumptions):
- src/data_engine/*.py
- src/kpi/generic_kpis.py
- dashboard/components/graceful.py
- dashboard/pages/0_upload_data.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# =====================================================================
# Hardcoded column name detection
# =====================================================================

# Column names from the NovaMart demo schema that should NEVER appear
# as hardcoded references in the dataset-agnostic layer
NOVAMARAT_COLUMNS = [
    'df["Revenue"]',
    'df["Date"]',
    'df["New_Customers"]',
    'df["Promotion_Spend"]',
    'df["Average_Price"]',
    'df["Holiday"]',
    'df["Competitor_Index"]',
    'df["Economic_Index"]',
    "df['Revenue']",
    "df['Date']",
    "df['New_Customers']",
    "df['Promotion_Spend']",
    "df['Average_Price']",
    "df['Holiday']",
    "df['Competitor_Index']",
    "df['Economic_Index']",
]

# Files in the dataset-agnostic layer that must NOT have hardcoded columns
AGNOSTIC_FILES = [
    ROOT / "src" / "data_engine" / "profiler.py",
    ROOT / "src" / "data_engine" / "column_classifier.py",
    ROOT / "src" / "data_engine" / "data_standardizer.py",
    ROOT / "src" / "data_engine" / "capability_engine.py",
    ROOT / "src" / "data_engine" / "schemas.py",
    ROOT / "src" / "kpi" / "generic_kpis.py",
    ROOT / "dashboard" / "components" / "graceful.py",
    ROOT / "dashboard" / "pages" / "0_upload_data.py",
]


def test_no_hardcoded_novamarat_columns_in_agnostic_layer():
    """Dataset-agnostic modules must not reference NovaMart column names directly."""
    violations = []
    for filepath in AGNOSTIC_FILES:
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        for pattern in NOVAMARAT_COLUMNS:
            if pattern in content:
                violations.append(f"{filepath.name}: found {pattern!r}")

    assert not violations, (
        f"Hardcoded NovaMart column references in dataset-agnostic code:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# =====================================================================
# Hardcoded file path detection
# =====================================================================

HARDCODED_PATHS = [
    "data/processed/marketing_data_clean.csv",
    "marketing_data_clean.csv",
    "data/processed/revenue_forecast.csv",
]


def test_no_hardcoded_demo_paths_in_agnostic_layer():
    """Dataset-agnostic modules must not reference the demo data directory."""
    violations = []
    for filepath in AGNOSTIC_FILES:
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        for path_str in HARDCODED_PATHS:
            if path_str in content:
                violations.append(f"{filepath.name}: found {path_str!r}")

    assert not violations, (
        f"Hardcoded demo file paths in dataset-agnostic code:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# =====================================================================
# Global model directory detection
# =====================================================================

def test_no_global_model_dir_in_agnostic_layer():
    """Dataset-agnostic modules must not reference a global models/ directory."""
    violations = []
    for filepath in AGNOSTIC_FILES:
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        # Look for patterns like ROOT / "models" or MODELS_DIR
        if re.search(r'ROOT\s*/\s*["\']models["\']', content):
            violations.append(f"{filepath.name}: references ROOT / 'models'")
        if "MODELS_DIR" in content and "# " not in content.split("MODELS_DIR")[0].split("\n")[-1]:
            # Allow MODELS_DIR in comments
            for line in content.split("\n"):
                if "MODELS_DIR" in line and not line.strip().startswith("#"):
                    violations.append(f"{filepath.name}: references MODELS_DIR")
                    break

    assert not violations, (
        f"Global model directory references in dataset-agnostic code:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# =====================================================================
# Hardcoded channel list detection
# =====================================================================

NOVAMARAT_CHANNEL_LISTS = [
    '["TV", "Radio", "Print", "Search"',
    "['TV', 'Radio', 'Print', 'Search'",
    '"TV_Spend"',
    '"Radio_Spend"',
    '"Print_Spend"',
    '"Search_Spend"',
    '"Social_Media_Spend"',
    '"Display_Ads_Spend"',
    '"Email_Marketing_Spend"',
    '"Influencer_Marketing_Spend"',
]


def test_no_hardcoded_channel_lists_in_agnostic_layer():
    """Dataset-agnostic modules must not contain hardcoded NovaMart channel lists."""
    violations = []
    for filepath in AGNOSTIC_FILES:
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        for pattern in NOVAMARAT_CHANNEL_LISTS:
            if pattern in content:
                violations.append(f"{filepath.name}: found {pattern!r}")

    assert not violations, (
        f"Hardcoded NovaMart channel lists in dataset-agnostic code:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# =====================================================================
# Currency symbol detection (should be parameterized, not hardcoded)
# =====================================================================

def test_no_hardcoded_currency_in_generic_kpis():
    """Generic KPIs should not hardcode any currency symbol."""
    kpi_file = ROOT / "src" / "kpi" / "generic_kpis.py"
    if not kpi_file.exists():
        return
    content = kpi_file.read_text(encoding="utf-8")
    # Check for common hardcoded currency markers
    currency_patterns = ["₹", "\\$", "€", "£"]
    violations = []
    for sym in currency_patterns:
        if sym in content:
            # Allow in comments/docstrings
            for i, line in enumerate(content.split("\n"), 1):
                stripped = line.strip()
                if sym in stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
                    violations.append(f"generic_kpis.py line {i}: hardcoded currency '{sym}'")

    assert not violations, (
        f"Hardcoded currency symbols in generic_kpis.py:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# =====================================================================
# analysis_runner.py — partial check (it bridges agnostic + legacy)
# =====================================================================

def test_analysis_runner_uses_data_dir_parameter():
    """analysis_runner.py must accept and use a data_dir / dataset_id
    parameter rather than a hardcoded path."""
    runner_file = ROOT / "src" / "data_engine" / "analysis_runner.py"
    if not runner_file.exists():
        return
    content = runner_file.read_text(encoding="utf-8")
    assert "dataset_id" in content, \
        "analysis_runner.py must accept dataset_id parameter"


# =====================================================================
# Known hardcoded legacy files (documented, not banned)
# These tests simply DOCUMENT where hardcoding exists in legacy code
# so Phase 1+ knows exactly what to fix.
# =====================================================================

def test_document_legacy_hardcoded_files():
    """Document (not fail on) files that contain known hardcoded references.
    This test always passes but prints a summary."""
    legacy_files = [
        ROOT / "src" / "models" / "marketing_mix_model.py",
        ROOT / "src" / "forecasting" / "forecast.py",
        ROOT / "src" / "optimization" / "budget_optimizer.py",
        ROOT / "src" / "optimization" / "simulator.py",
        ROOT / "src" / "ai" / "analyst.py",
        ROOT / "dashboard" / "components" / "data_loader.py",
        ROOT / "api" / "main.py",
    ]
    known_hardcoded = []
    for filepath in legacy_files:
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        hardcoded_items = []
        if "marketing_data_clean.csv" in content:
            hardcoded_items.append("marketing_data_clean.csv")
        if 'df["Revenue"]' in content or "df['Revenue']" in content:
            hardcoded_items.append('df["Revenue"]')
        if 'df["Date"]' in content or "df['Date']" in content:
            hardcoded_items.append('df["Date"]')
        for ch in ["TV", "Radio", "Print", "Search", "Social_Media"]:
            if f'"{ch}_Spend"' in content or f"'{ch}_Spend'" in content:
                hardcoded_items.append(f"{ch}_Spend")
                break
        if hardcoded_items:
            known_hardcoded.append((filepath.name, hardcoded_items))

    # This test ALWAYS passes — it's documentation, not enforcement
    # The output will be visible in verbose test runs
    if known_hardcoded:
        print("\n--- Known hardcoded legacy files (to be fixed in Phase 1+) ---")
        for fname, items in known_hardcoded:
            print(f"  {fname}: {', '.join(items)}")
    assert True  # Always passes


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
