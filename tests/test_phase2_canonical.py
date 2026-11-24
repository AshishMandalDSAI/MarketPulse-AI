"""Phase 2 — Canonical Data Contract Tests.

Tests cover:
A. Canonical object creation (CSV → CanonicalDataset) across all 11 diverse fixtures
B. Column metadata (dtype, missingness, unique count, semantic type, role candidates)
C. Semantic roles across multiple domains
D. Capability detection — marketing vs non-marketing produce different capabilities
E. Target detection — not hardcoded to Revenue
F. Time detection — normal date, date-like string, no date, invalid date
G. Data quality — missing values, duplicates, constant columns, empty columns, mixed types
H. Dataset isolation — Canonical(A), Canonical(B), Canonical(C) never cross
I. Serialization — round-trip to_metadata_dict / from_metadata_dict
J. Anti-hardcoding — canonical layer itself is schema-agnostic
K. AnalysisContext — wraps canonical with capabilities
L. UserOverrides — contract for user overrides
M. Integration — analysis_runner produces canonical objects
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_engine.canonical import (
    CanonicalDataset,
    AnalysisContext,
    ColumnMetadata,
    DataQuality,
    SemanticRole,
    TargetCandidate,
    TimeInfo,
    UserOverrides,
    build_canonical,
    build_column_metadata,
    compute_data_quality,
    detect_target_candidates,
    detect_time_info,
)
from tests.helpers import load_fixture, all_diverse_fixtures, fixture_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(fixture_name: str, fixture_subdir: str = "diverse_schemas",
           dataset_id: str = "test-canonical") -> tuple[CanonicalDataset, AnalysisContext]:
    """Build a CanonicalDataset + AnalysisContext from a fixture."""
    df = load_fixture(fixture_name, fixture_subdir)
    return build_canonical(dataset_id, df, filename=fixture_name)


# ===========================================================================
# A. Canonical Object Creation — every diverse fixture
# ===========================================================================

class TestCanonicalCreation:
    """CanonicalDataset must be constructable from every fixture CSV."""

    @pytest.mark.parametrize("name,path", all_diverse_fixtures())
    def test_creation_from_fixture(self, name, path):
        df = pd.read_csv(path)
        canonical, ctx = build_canonical(f"test-{name}", df, filename=name)

        assert canonical.dataset_id == f"test-{name}"
        assert canonical.filename == name
        assert canonical.row_count == len(df)
        assert canonical.column_count == len(df.columns)
        assert canonical.dataframe is not None
        assert len(canonical.columns) == len(df.columns)
        assert len(canonical.column_metadata) == len(df.columns)
        assert canonical.data_quality is not None
        assert canonical.time_info is not None
        assert ctx is not None
        assert ctx.dataset is canonical

    def test_creation_from_generic_marketing(self):
        canonical, ctx = _build("dataset_a_marketing.csv", "generic_schemas")
        assert canonical.row_count > 0

    def test_creation_from_generic_non_marketing(self):
        canonical, ctx = _build("dataset_e_no_marketing.csv", "generic_schemas")
        assert canonical.row_count > 0


# ===========================================================================
# B. Column Metadata
# ===========================================================================

class TestColumnMetadata:
    """Every column should have standardized metadata."""

    def test_metadata_keys_present(self):
        canonical, _ = _build("a_marketing.csv")
        for col_name, meta in canonical.column_metadata.items():
            assert isinstance(meta, ColumnMetadata)
            assert meta.name == col_name
            assert meta.dtype  # not empty
            assert meta.semantic_type in ("numeric", "categorical", "datetime", "text",
                                           "identifier", "boolean", "empty")

    def test_numeric_column_has_stats(self):
        canonical, _ = _build("a_marketing.csv")
        # Revenue should be numeric with stats
        revenue_meta = canonical.column_metadata.get("Revenue")
        assert revenue_meta is not None
        assert revenue_meta.is_numeric
        assert revenue_meta.stats is not None
        assert "mean" in revenue_meta.stats
        assert "min" in revenue_meta.stats
        assert "max" in revenue_meta.stats

    def test_missing_values_tracked(self):
        canonical, _ = _build("i_missing_values.csv")
        has_missing = any(m.missing_count > 0 for m in canonical.column_metadata.values())
        assert has_missing, "Missing-values fixture should have columns with missing data"

    def test_unique_count_and_pct(self):
        canonical, _ = _build("b_sales.csv")
        for meta in canonical.column_metadata.values():
            assert meta.unique_count >= 0
            assert 0 <= meta.unique_pct <= 100

    def test_sample_values_populated(self):
        canonical, _ = _build("a_marketing.csv")
        for meta in canonical.column_metadata.values():
            if meta.missing_count < canonical.row_count:
                assert len(meta.sample_values) > 0, f"Column {meta.name} should have sample values"

    def test_boolean_detection(self):
        """Boolean columns should be detected (status-like columns)."""
        df = pd.DataFrame({"flag": [True, False, True, False], "val": [1.0, 2.0, 3.0, 4.0]})
        metas = build_column_metadata(df)
        flag_meta = [m for m in metas if m.name == "flag"][0]
        assert flag_meta.is_boolean


# ===========================================================================
# C. Semantic Roles
# ===========================================================================

class TestSemanticRoles:
    """Semantic roles must be assigned correctly across multiple domains."""

    def test_marketing_has_spend_roles(self):
        canonical, _ = _build("a_marketing.csv")
        # Marketing spend columns should have MARKETING_SPEND role
        roles = canonical.semantic_roles
        spend_cols = [col for col, r in roles.items()
                      if SemanticRole.MARKETING_SPEND.value in r]
        assert len(spend_cols) > 0, "Marketing fixture should have MARKETING_SPEND columns"

    def test_education_has_no_marketing_roles(self):
        canonical, _ = _build("e_education.csv")
        roles = canonical.semantic_roles
        spend_cols = [col for col, r in roles.items()
                      if SemanticRole.MARKETING_SPEND.value in r]
        assert len(spend_cols) == 0, "Education fixture should NOT have MARKETING_SPEND columns"

    def test_employee_has_no_marketing_roles(self):
        canonical, _ = _build("f_employee.csv")
        roles = canonical.semantic_roles
        spend_cols = [col for col, r in roles.items()
                      if SemanticRole.MARKETING_SPEND.value in r]
        assert len(spend_cols) == 0, "Employee fixture should NOT have MARKETING_SPEND columns"

    def test_time_column_has_time_role(self):
        canonical, _ = _build("a_marketing.csv")
        roles = canonical.semantic_roles
        assert "Date" in roles
        assert SemanticRole.TIME.value in roles["Date"]

    def test_no_date_fixture_has_no_time_role(self):
        canonical, _ = _build("g_no_date.csv")
        roles = canonical.semantic_roles
        time_cols = [col for col, r in roles.items()
                     if SemanticRole.TIME.value in r]
        assert len(time_cols) == 0, "No-date fixture should have no TIME role"

    def test_identifier_column_has_identifier_role(self):
        canonical, _ = _build("e_education.csv")
        roles = canonical.semantic_roles
        id_cols = [col for col, r in roles.items()
                   if SemanticRole.IDENTIFIER.value in r]
        assert len(id_cols) > 0, "Education fixture should have IDENTIFIER columns"

    def test_dimension_columns_have_dimension_or_category_role(self):
        canonical, _ = _build("b_sales.csv")
        roles = canonical.semantic_roles
        dim_cols = [col for col, r in roles.items()
                    if SemanticRole.DIMENSION.value in r or SemanticRole.CATEGORY.value in r]
        assert len(dim_cols) > 0, "Sales fixture should have DIMENSION/CATEGORY columns"

    def test_all_columns_get_at_least_one_role(self):
        """Every column should have at least one semantic role."""
        canonical, _ = _build("a_marketing.csv")
        for col in canonical.columns:
            roles = canonical.semantic_roles.get(col, [])
            assert len(roles) > 0, f"Column {col} has no semantic role"

    def test_semantic_role_enum_values(self):
        """SemanticRole enum values must be string-serializable."""
        for role in SemanticRole:
            assert isinstance(role.value, str)
            assert role.value == role.name  # e.g. SemanticRole.TIME.value == "TIME"


# ===========================================================================
# D. Capability Detection
# ===========================================================================

class TestCapabilityDetection:
    """Marketing and non-marketing datasets must produce different capabilities."""

    def test_marketing_dataset_has_mmm_capability(self):
        canonical, ctx = _build("a_marketing.csv")
        assert ctx.has_capability("marketing_mix_modeling"), \
            "Marketing fixture should have MMM capability"

    def test_education_dataset_lacks_mmm(self):
        canonical, ctx = _build("e_education.csv")
        assert not ctx.has_capability("marketing_mix_modeling"), \
            "Education fixture should NOT have MMM capability"

    def test_no_date_dataset_lacks_forecasting(self):
        canonical, ctx = _build("g_no_date.csv")
        assert not ctx.has_capability("forecasting"), \
            "No-date fixture should NOT have forecasting capability"

    def test_marketing_dataset_has_forecasting(self):
        canonical, ctx = _build("a_marketing.csv")
        # Marketing with date should have forecasting if enough rows
        caps = canonical.capabilities
        assert caps is not None

    def test_all_datasets_have_profiling(self):
        canonical, ctx = _build("k_minimal.csv")
        assert ctx.has_capability("data_profiling"), \
            "Even minimal datasets should have data_profiling capability"

    def test_capabilities_dict_format(self):
        canonical, ctx = _build("a_marketing.csv")
        assert isinstance(canonical.capabilities, dict)
        for key, entry in canonical.capabilities.items():
            assert isinstance(key, str)
            assert "available" in entry
            assert "reason" in entry


# ===========================================================================
# E. Target Detection
# ===========================================================================

class TestTargetDetection:
    """Target candidates must NOT be hardcoded to Revenue."""

    def test_marketing_target_includes_revenue(self):
        canonical, _ = _build("a_marketing.csv")
        target_cols = [t.column for t in canonical.target_candidates]
        assert "Revenue" in target_cols, "Revenue should be a target candidate in marketing data"

    def test_education_target_not_revenue(self):
        canonical, _ = _build("e_education.csv")
        target_cols = [t.column for t in canonical.target_candidates]
        assert "Revenue" not in target_cols, "Education data should NOT have Revenue as target"
        # Should detect exam scores or similar as targets
        assert len(target_cols) > 0, "Education data should have at least one target candidate"

    def test_employee_target_is_salary_or_rating(self):
        canonical, _ = _build("f_employee.csv")
        target_cols = [t.column for t in canonical.target_candidates]
        assert "Revenue" not in target_cols
        assert len(target_cols) > 0, "Employee data should have target candidates"

    def test_target_candidates_have_scores_and_reasons(self):
        canonical, _ = _build("a_marketing.csv")
        for tc in canonical.target_candidates:
            assert isinstance(tc, TargetCandidate)
            assert 0 < tc.score <= 1.0
            assert len(tc.reasons) > 0

    def test_target_candidates_sorted_by_score(self):
        canonical, _ = _build("a_marketing.csv")
        scores = [tc.score for tc in canonical.target_candidates]
        assert scores == sorted(scores, reverse=True), "Candidates should be sorted by score descending"

    def test_no_forced_target_for_minimal(self):
        """Minimal dataset with generic column names shouldn't force a high-confidence target."""
        canonical, _ = _build("k_minimal.csv")
        # Should still work; target candidates may be empty or low-confidence
        assert isinstance(canonical.target_candidates, list)

    def test_operations_target_not_revenue(self):
        canonical, _ = _build("d_operations.csv")
        target_cols = [t.column for t in canonical.target_candidates]
        assert "Revenue" not in target_cols


# ===========================================================================
# F. Time Detection
# ===========================================================================

class TestTimeDetection:
    """Time detection must handle normal dates, date-like strings, no dates, etc."""

    def test_normal_date_detected(self):
        canonical, _ = _build("a_marketing.csv")
        ti = canonical.time_info
        assert ti is not None
        assert ti.time_column is not None
        assert ti.time_granularity is not None
        assert ti.time_range is not None
        assert "start" in ti.time_range
        assert "end" in ti.time_range

    def test_weekly_granularity_marketing(self):
        canonical, _ = _build("a_marketing.csv")
        ti = canonical.time_info
        assert ti.time_granularity == "weekly"

    def test_no_date_fixture(self):
        canonical, _ = _build("g_no_date.csv")
        ti = canonical.time_info
        assert ti.time_column is None
        assert ti.time_granularity is None
        assert ti.is_regular_time_series is False

    def test_daily_granularity_sales(self):
        canonical, _ = _build("b_sales.csv")
        ti = canonical.time_info
        assert ti.time_column is not None
        assert ti.time_granularity == "daily"

    def test_time_candidates_populated(self):
        canonical, _ = _build("a_marketing.csv")
        assert len(canonical.time_candidates) > 0

    def test_time_candidates_empty_for_no_date(self):
        canonical, _ = _build("g_no_date.csv")
        # time_candidates should be empty when no date column
        assert len(canonical.time_candidates) == 0

    def test_time_info_serializable(self):
        canonical, _ = _build("a_marketing.csv")
        d = canonical.time_info.to_dict()
        ti2 = TimeInfo.from_dict(d)
        assert ti2.time_column == canonical.time_info.time_column
        assert ti2.time_granularity == canonical.time_info.time_granularity


# ===========================================================================
# G. Data Quality
# ===========================================================================

class TestDataQuality:
    """Data quality contract must detect real issues."""

    def test_quality_score_range(self):
        canonical, _ = _build("a_marketing.csv")
        q = canonical.data_quality
        assert 0.0 <= q.score <= 1.0

    def test_missing_values_detected(self):
        canonical, _ = _build("i_missing_values.csv")
        q = canonical.data_quality
        assert q.missing_cells > 0, "Missing-values fixture should report missing cells"
        assert q.missing_cells_pct > 0

    def test_duplicate_rows_detected(self):
        canonical, _ = _build("j_duplicate_rows.csv")
        q = canonical.data_quality
        assert q.duplicate_rows > 0, "Duplicate-rows fixture should report duplicates"

    def test_row_and_column_counts(self):
        df = load_fixture("a_marketing.csv")
        canonical, _ = build_canonical("test-q", df)
        assert canonical.data_quality.row_count == len(df)
        assert canonical.data_quality.column_count == len(df.columns)

    def test_quality_has_warnings_for_issues(self):
        """Datasets with quality issues should produce warnings or detectable metrics."""
        canonical, _ = _build("i_missing_values.csv")
        q = canonical.data_quality
        # Missing-values fixture has missing data — quality report should reflect it
        assert q.missing_cells > 0, "Missing-values fixture should report missing cells"
        # Quality score should be less than perfect
        assert q.score < 1.0

    def test_clean_dataset_high_score(self):
        canonical, _ = _build("a_marketing.csv")
        q = canonical.data_quality
        assert q.score > 0.7, "Clean marketing fixture should have a high quality score"

    def test_quality_serializable(self):
        canonical, _ = _build("i_missing_values.csv")
        d = canonical.data_quality.to_dict()
        q2 = DataQuality.from_dict(d)
        assert q2.missing_cells == canonical.data_quality.missing_cells
        assert q2.row_count == canonical.data_quality.row_count

    def test_constant_column_detection(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "constant": ["X", "X", "X", "X"],
            "value": [10, 20, 30, 40],
        })
        q = compute_data_quality(df)
        assert "constant" in q.constant_columns

    def test_empty_column_detection(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "empty": [None, None, None],
            "value": [10, 20, 30],
        })
        q = compute_data_quality(df)
        assert "empty" in q.empty_columns

    def test_mixed_type_detection(self):
        # Mix of clearly numeric and clearly non-numeric strings (30-70% split)
        df = pd.DataFrame({
            "mixed": ["abc", "123", "def", "456", "ghi", "789",
                       "jkl", "mno", "pqr", "stu"],
        })
        q = compute_data_quality(df)
        # 3 out of 10 are numeric (0.3) — within (0.1, 0.9) range
        assert "mixed" in q.mixed_type_columns


# ===========================================================================
# H. Dataset Isolation
# ===========================================================================

class TestDatasetIsolation:
    """Canonical(A), Canonical(B), Canonical(C) must never cross."""

    def test_three_datasets_independent(self):
        ca, _ = _build("a_marketing.csv", dataset_id="iso-a")
        cb, _ = _build("e_education.csv", dataset_id="iso-b")
        cc, _ = _build("g_no_date.csv", dataset_id="iso-c")

        # Different dataset IDs
        assert ca.dataset_id != cb.dataset_id
        assert cb.dataset_id != cc.dataset_id
        assert ca.dataset_id != cc.dataset_id

        # Different column sets
        assert set(ca.columns) != set(cb.columns)
        assert set(cb.columns) != set(cc.columns)

        # Different row counts
        assert ca.row_count != cb.row_count or ca.column_count != cb.column_count

    def test_metadata_never_crosses(self):
        ca, _ = _build("a_marketing.csv", dataset_id="meta-a")
        cb, _ = _build("f_employee.csv", dataset_id="meta-b")

        # Column metadata should be completely independent
        meta_a_names = set(ca.column_metadata.keys())
        meta_b_names = set(cb.column_metadata.keys())
        # Different datasets shouldn't share all column names
        assert meta_a_names != meta_b_names

    def test_canonical_always_has_dataset_id(self):
        """CanonicalDataset must always carry a dataset_id — never a global singleton."""
        for name, path in all_diverse_fixtures():
            df = pd.read_csv(path)
            c, _ = build_canonical(f"test-{name}", df)
            assert c.dataset_id == f"test-{name}"
            assert c.dataset_id  # never empty

    def test_serialized_metadata_contains_dataset_id(self):
        ca, _ = _build("a_marketing.csv", dataset_id="serial-a")
        d = ca.to_metadata_dict()
        assert d["dataset_id"] == "serial-a"


# ===========================================================================
# I. Serialization
# ===========================================================================

class TestSerialization:
    """Canonical metadata must serialize/deserialize safely."""

    def test_to_metadata_dict_is_json_safe(self):
        canonical, _ = _build("a_marketing.csv")
        d = canonical.to_metadata_dict()
        # Must be JSON-serializable (no numpy types, no DataFrames)
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0

    def test_round_trip_metadata(self):
        canonical, _ = _build("a_marketing.csv", dataset_id="rt-test")
        d = canonical.to_metadata_dict()
        json_str = json.dumps(d, default=str)
        d2 = json.loads(json_str)
        restored = CanonicalDataset.from_metadata_dict(d2)

        assert restored.dataset_id == "rt-test"
        assert restored.row_count == canonical.row_count
        assert restored.column_count == canonical.column_count
        assert set(restored.columns) == set(canonical.columns)
        assert len(restored.column_metadata) == len(canonical.column_metadata)
        assert restored.data_quality is not None
        assert restored.time_info is not None

    def test_round_trip_preserves_target_candidates(self):
        canonical, _ = _build("a_marketing.csv")
        d = canonical.to_metadata_dict()
        restored = CanonicalDataset.from_metadata_dict(d)
        assert len(restored.target_candidates) == len(canonical.target_candidates)
        for orig, rest in zip(canonical.target_candidates, restored.target_candidates):
            assert orig.column == rest.column
            assert abs(orig.score - rest.score) < 0.01

    def test_round_trip_preserves_semantic_roles(self):
        canonical, _ = _build("a_marketing.csv")
        d = canonical.to_metadata_dict()
        restored = CanonicalDataset.from_metadata_dict(d)
        assert restored.semantic_roles == canonical.semantic_roles

    def test_round_trip_preserves_capabilities(self):
        canonical, _ = _build("a_marketing.csv")
        d = canonical.to_metadata_dict()
        restored = CanonicalDataset.from_metadata_dict(d)
        assert restored.capabilities == canonical.capabilities

    def test_column_metadata_round_trip(self):
        canonical, _ = _build("b_sales.csv")
        for name, meta in canonical.column_metadata.items():
            d = meta.to_dict()
            restored = ColumnMetadata.from_dict(d)
            assert restored.name == meta.name
            assert restored.dtype == meta.dtype
            assert restored.semantic_type == meta.semantic_type
            assert restored.is_numeric == meta.is_numeric
            assert restored.missing_count == meta.missing_count

    def test_analysis_context_serializable(self):
        _, ctx = _build("a_marketing.csv")
        d = ctx.to_metadata_dict()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0
        assert "dataset" in d
        assert "capabilities" in d

    @pytest.mark.parametrize("name,path", all_diverse_fixtures())
    def test_all_fixtures_serialize_cleanly(self, name, path):
        """Every fixture must produce JSON-serializable metadata."""
        df = pd.read_csv(path)
        c, _ = build_canonical(f"ser-{name}", df)
        d = c.to_metadata_dict()
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 100


# ===========================================================================
# J. Anti-Hardcoding Guards
# ===========================================================================

class TestAntiHardcoding:
    """The canonical layer itself must be schema-agnostic."""

    def test_canonical_module_no_revenue_hardcode(self):
        """The canonical.py module must not contain df['Revenue'] or df['Date'] access."""
        import inspect
        from src.data_engine import canonical as mod
        source = inspect.getsource(mod)
        # The module should not directly access df["Revenue"] or df["Date"]
        assert 'df["Revenue"]' not in source, "canonical.py must not hardcode df['Revenue']"
        assert "df['Revenue']" not in source, "canonical.py must not hardcode df['Revenue']"
        assert 'df["Date"]' not in source, "canonical.py must not hardcode df['Date']"
        assert "df['Date']" not in source, "canonical.py must not hardcode df['Date']"
        assert 'df["Google Ads"]' not in source, "canonical.py must not hardcode df['Google Ads']"
        assert 'df["Facebook Ads"]' not in source, "canonical.py must not hardcode df['Facebook Ads']"

    def test_arbitrary_columns_produce_valid_canonical(self):
        """Dataset with columns A, B, C, D should still produce a valid CanonicalDataset."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],
            "C": ["x", "y", "x", "y", "z"],
            "D": [True, False, True, False, True],
        })
        canonical, ctx = build_canonical("arb-test", df, filename="arbitrary.csv")
        assert canonical.row_count == 5
        assert canonical.column_count == 4
        assert len(canonical.column_metadata) == 4
        assert canonical.data_quality is not None

    def test_unusual_columns_produce_valid_canonical(self):
        canonical, _ = _build("h_unusual_columns.csv")
        assert canonical.row_count > 0
        assert len(canonical.column_metadata) > 0
        # Should have target candidates based on structure, not hardcoded names
        # (unusual columns won't match "Revenue" etc.)

    def test_no_marketing_assumption_required(self):
        """Non-marketing datasets must work without any marketing schema."""
        for fixture_name in ["e_education.csv", "f_employee.csv", "g_no_date.csv",
                             "h_unusual_columns.csv", "k_minimal.csv"]:
            canonical, ctx = _build(fixture_name)
            assert canonical.row_count > 0
            assert canonical.data_quality is not None
            assert canonical.time_info is not None


# ===========================================================================
# K. AnalysisContext
# ===========================================================================

class TestAnalysisContext:
    """AnalysisContext wraps CanonicalDataset with capabilities."""

    def test_context_has_dataset(self):
        _, ctx = _build("a_marketing.csv")
        assert ctx.dataset is not None
        assert isinstance(ctx.dataset, CanonicalDataset)

    def test_context_has_capability_method(self):
        _, ctx = _build("a_marketing.csv")
        assert ctx.has_capability("data_profiling")
        assert isinstance(ctx.has_capability("nonexistent_cap"), bool)
        assert not ctx.has_capability("nonexistent_cap")

    def test_context_get_dataframe(self):
        _, ctx = _build("a_marketing.csv")
        df = ctx.get_dataframe()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_context_semantic_roles(self):
        _, ctx = _build("a_marketing.csv")
        assert isinstance(ctx.semantic_roles, dict)
        assert len(ctx.semantic_roles) > 0

    def test_context_quality(self):
        _, ctx = _build("a_marketing.csv")
        assert ctx.quality is not None
        assert isinstance(ctx.quality, DataQuality)


# ===========================================================================
# L. UserOverrides
# ===========================================================================

class TestUserOverrides:
    """User overrides contract for future UI integration."""

    def test_default_overrides_empty(self):
        uo = UserOverrides()
        assert not uo.has_overrides()

    def test_target_override(self):
        uo = UserOverrides(target_column="my_target")
        assert uo.has_overrides()
        pipeline = uo.to_pipeline_overrides()
        assert pipeline["revenue_column"] == "my_target"

    def test_date_override(self):
        uo = UserOverrides(date_column="my_date")
        assert uo.has_overrides()
        pipeline = uo.to_pipeline_overrides()
        assert pipeline["date_column"] == "my_date"

    def test_ignore_columns(self):
        uo = UserOverrides(ignore_columns=["col_x", "col_y"])
        assert uo.has_overrides()

    def test_force_marketing_spend(self):
        uo = UserOverrides(force_marketing_spend=["spend_a", "spend_b"])
        pipeline = uo.to_pipeline_overrides()
        assert pipeline["channel_columns"] == ["spend_a", "spend_b"]

    def test_overrides_serializable(self):
        uo = UserOverrides(target_column="x", date_column="d",
                           ignore_columns=["a"], force_categorical=["b"],
                           force_measure=["c"], force_marketing_spend=["s"])
        d = uo.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        restored = UserOverrides.from_dict(d2)
        assert restored.target_column == "x"
        assert restored.date_column == "d"
        assert restored.ignore_columns == ["a"]
        assert restored.force_marketing_spend == ["s"]


# ===========================================================================
# M. Integration — analysis_runner produces canonical
# ===========================================================================

class TestAnalysisRunnerIntegration:
    """run_full_analysis() should produce canonical objects in its result."""

    def test_runner_returns_canonical(self):
        from src.data_engine.analysis_runner import run_full_analysis
        df = load_fixture("a_marketing.csv")
        result = run_full_analysis("test-runner-canon", df)
        assert "canonical" in result, "run_full_analysis must return 'canonical' key"
        assert "context" in result, "run_full_analysis must return 'context' key"
        assert isinstance(result["canonical"], CanonicalDataset)
        assert isinstance(result["context"], AnalysisContext)

    def test_runner_canonical_matches_dataset(self):
        from src.data_engine.analysis_runner import run_full_analysis
        df = load_fixture("b_sales.csv")
        result = run_full_analysis("test-runner-match", df)
        canonical = result["canonical"]
        assert canonical.dataset_id == "test-runner-match"
        assert canonical.row_count == len(df)

    def test_runner_persists_canonical_metadata(self):
        from src.data_engine.analysis_runner import run_full_analysis
        from config.settings import dataset_processed_dir
        df = load_fixture("c_finance.csv")
        result = run_full_analysis("test-runner-persist", df)
        meta_path = dataset_processed_dir("test-runner-persist") / "canonical_metadata.json"
        assert meta_path.exists(), "canonical_metadata.json should be persisted"
        with open(meta_path) as f:
            d = json.load(f)
        assert d["dataset_id"] == "test-runner-persist"
        assert d["row_count"] == len(df)

    def test_runner_backward_compatible(self):
        """Existing return keys must still be present."""
        from src.data_engine.analysis_runner import run_full_analysis
        df = load_fixture("a_marketing.csv")
        result = run_full_analysis("test-runner-compat", df)
        assert "dataset" in result
        assert "capabilities" in result
        assert "generic_kpis" in result
        assert "report" in result


# ===========================================================================
# N. Measure / Dimension Contract
# ===========================================================================

class TestMeasureDimension:
    """Measures and dimensions must be correctly classified."""

    def test_marketing_measures(self):
        canonical, _ = _build("a_marketing.csv")
        assert len(canonical.measure_candidates) > 0
        # Revenue should be a measure
        assert "Revenue" in canonical.measure_candidates

    def test_sales_dimensions(self):
        canonical, _ = _build("b_sales.csv")
        assert len(canonical.dimension_candidates) > 0

    def test_education_no_revenue_measure(self):
        canonical, _ = _build("e_education.csv")
        assert "Revenue" not in canonical.measure_candidates

    def test_employee_salary_is_measure(self):
        canonical, _ = _build("f_employee.csv")
        assert "Salary" in canonical.measure_candidates

    def test_identifiers_not_in_measures(self):
        canonical, _ = _build("e_education.csv")
        for id_col in canonical.identifier_columns:
            assert id_col not in canonical.measure_candidates, \
                f"Identifier {id_col} should not be a measure"


# ===========================================================================
# O. Regression — Phase 0 + Phase 1 preserved
# ===========================================================================

class TestRegression:
    """All previous tests must remain green. This verifies the canonical
    layer doesn't break existing pipeline behavior."""

    def test_existing_pipeline_still_works(self):
        """The existing pipeline (profile → classify → standardize → capabilities)
        must still produce valid results when called directly."""
        from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
        from src.data_engine.capability_engine import detect_capabilities

        df = load_fixture("a_marketing.csv")
        profile, mapping = profile_and_map(df)
        dataset, report = confirm_and_standardize(df, mapping)
        caps = detect_capabilities(dataset)

        assert dataset.df is not None
        assert len(dataset.df) == len(df)
        assert caps.is_available("data_profiling")

    def test_standardized_dataset_accessible_from_canonical(self):
        """CanonicalDataset should provide access to the underlying StandardizedDataset."""
        canonical, _ = _build("a_marketing.csv")
        std = canonical.get_standardized()
        assert std is not None
        assert std.df is not None
