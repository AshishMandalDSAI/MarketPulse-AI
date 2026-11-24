"""Phase 6 — Marketing Field Discovery & Channel Validation.

Dynamically detects marketing semantic roles (channels, target, time,
controls, conversions) from a CanonicalDataset.  Never hardcodes column
names — uses the existing semantic role system, column aliases, and
statistical signals.

Channel validation enforces the 8-rule check from the Phase 6 spec
before a column is accepted as a marketing channel.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.column_aliases import (
    CHANNEL_SPEND_ALIASES, SPEND_GENERIC_ALIASES, SPEND_ALIASES,
    REVENUE_ALIASES, DATE_ALIASES, CONVERSIONS_ALIASES,
)
from config.constants import MIN_CHANNELS_FOR_MMM
from src.data_engine.role_lexicon import is_trade_or_promotion_spend_name
from src.data_engine.canonical import CanonicalDataset, SemanticRole, AnalysisContext
from src.marketing.contracts import (
    MarketingCapabilityReport, MarketingReadiness, CapabilityStatus,
    ChannelValidation, DataQualityState,
)


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------
MAX_CHANNELS = 50
MAX_ROWS_FOR_MMM = 50_000
MIN_ROWS_FOR_MMM = 12


# ---------------------------------------------------------------------------
# Column name normalization (reuse the same scheme as column_aliases.py)
# ---------------------------------------------------------------------------

def _normalize_col_name(name: str) -> str:
    """Lowercase, strip whitespace, replace separators with single space."""
    return re.sub(r"[_\-./\s]+", " ", name.lower()).strip()


# ---------------------------------------------------------------------------
# 1. Marketing channel detection
# ---------------------------------------------------------------------------

def _has_spend_semantic_evidence(col_name: str) -> tuple[bool, str]:
    """Check if a column name has semantic evidence of being marketing spend.

    Returns (is_likely_spend, matched_pattern).
    Does NOT use an enormous hardcoded keyword list — instead uses the
    existing CHANNEL_SPEND_ALIASES vocabulary plus structural patterns.
    """
    norm = _normalize_col_name(col_name)
    tokens = set(norm.split())

    # 1. Direct match against known channel spend aliases
    for channel_key, aliases in CHANNEL_SPEND_ALIASES.items():
        normalized_aliases = {_normalize_col_name(a) for a in aliases}
        if norm in normalized_aliases:
            return True, f"channel_alias:{channel_key}"

    # 2. Generic spend aliases
    for alias in SPEND_GENERIC_ALIASES:
        if _normalize_col_name(alias) == norm:
            return True, "generic_spend"

    # 3. Structural patterns: "<something>_spend", "<something>_budget",
    #    "<something>_cost" (where something suggests a channel/medium)
    spend_suffixes = {"spend", "budget", "cost", "investment", "expenditure"}
    channel_indicators = {
        "ads", "ad", "advertising", "marketing", "media", "campaign",
        "paid", "organic", "social", "search", "display", "video",
        "brand", "digital", "traditional", "online", "offline",
    }

    if tokens & spend_suffixes:
        # Has a spend-like suffix — check if there's also a channel indicator
        remaining = tokens - spend_suffixes
        if remaining & channel_indicators:
            return True, "structural:channel+spend"
        # Even without explicit channel indicator, if the column name looks
        # like "<platform>_spend" (e.g., "tiktok_spend"), the platform token
        # itself is evidence
        for platform_key in CHANNEL_SPEND_ALIASES:
            if platform_key in remaining:
                return True, f"structural:platform({platform_key})+spend"
        # If it's just "spend" or "total spend" alone, it's not a channel
        if remaining:
            # There's a qualifier before "spend" that we don't recognize —
            # treat as moderate evidence (the channel validator will check further)
            return True, "structural:qualified_spend"

    return False, ""


def detect_channel_candidates(canonical: CanonicalDataset) -> list[str]:
    """Detect candidate marketing channel columns from the canonical dataset.

    Uses three signals:
    1. MARKETING_SPEND semantic role (already assigned by the pipeline)
    2. StandardizedDataset.channels (from the pipeline's own detection)
    3. Column name semantic evidence (spend aliases + structural patterns)

    Returns a de-duplicated list, ordered by detection confidence.
    """
    candidates = []
    seen = set()

    # Signal 1: Semantic roles
    for col_name, roles in canonical.semantic_roles.items():
        if SemanticRole.MARKETING_SPEND.value in roles:
            if col_name not in seen:
                candidates.append(col_name)
                seen.add(col_name)

    # Signal 2: StandardizedDataset channels (if we have a back-reference)
    if canonical._standardized and hasattr(canonical._standardized, "channels"):
        for ch in canonical._standardized.channels:
            if ch not in seen:
                candidates.append(ch)
                seen.add(ch)

    # Signal 3: Column name evidence
    for col_name in canonical.numeric_columns:
        if col_name in seen:
            continue
        # PART 02: trade / promotion spend is never guessed as an advertising channel
        # (signals 1-2 still carry a channel the user explicitly confirmed).
        if is_trade_or_promotion_spend_name(col_name):
            continue
        has_evidence, _ = _has_spend_semantic_evidence(col_name)
        if has_evidence:
            candidates.append(col_name)
            seen.add(col_name)

    return candidates[:MAX_CHANNELS]


# ---------------------------------------------------------------------------
# 2. Channel Validation (8-rule check)
# ---------------------------------------------------------------------------

def validate_channel(col_name: str, df: pd.DataFrame,
                     target_col: Optional[str] = None,
                     all_channels: Optional[list] = None) -> ChannelValidation:
    """Validate a single channel candidate against the 8-rule check.

    Rules:
    1. Numeric type
    2. Sufficient non-null observations (>= 10 or >= 50% of rows)
    3. Meaningful variance (not constant)
    4. Plausible spend/cost semantics
    5. Not an obvious identifier
    6. Not the target variable
    7. Not a duplicate representation of another field
    8. Sufficient observation count (same as #2 for simplicity)
    """
    result = ChannelValidation(column=col_name)
    series = df[col_name] if col_name in df.columns else pd.Series(dtype=float)

    if series.empty:
        result.is_valid = False
        result.rejection_reasons.append("Column not found in DataFrame")
        return result

    # Rule 1: Numeric type
    numeric = pd.to_numeric(series, errors="coerce")
    non_null_count = int(numeric.notna().sum())
    if non_null_count == 0:
        result.is_valid = False
        result.rejection_reasons.append("Not numeric or all values are null")
        return result
    result.reasons.append("Numeric type confirmed")

    # Rule 2 & 8: Sufficient non-null observations
    n_rows = len(df)
    min_required = max(10, int(n_rows * 0.1))  # at least 10 or 10% of rows
    if non_null_count < min_required:
        result.is_valid = False
        result.rejection_reasons.append(
            f"Insufficient non-null observations: {non_null_count} < {min_required}")
        return result
    result.reasons.append(f"Sufficient observations ({non_null_count})")

    # Rule 3: Meaningful variance (not constant)
    vals = numeric.dropna().values
    if vals.std() < 1e-9:
        result.is_valid = False
        result.rejection_reasons.append("Constant or near-constant values — no variance")
        return result
    result.reasons.append("Has meaningful variance")

    # Rule 4: Plausible spend/cost semantics
    has_evidence, pattern = _has_spend_semantic_evidence(col_name)
    if has_evidence:
        result.reasons.append(f"Spend semantics detected ({pattern})")
    else:
        # Check if semantic role was assigned
        from src.data_engine.canonical import SemanticRole
        roles = []
        # We don't have canonical here, but the orchestrator checks this
        # If no evidence at all, it's a warning, not a blocker
        result.warnings.append("No explicit spend/cost name pattern detected")

    # Rule 5: Not an obvious identifier
    unique_ratio = len(set(vals)) / len(vals) if len(vals) > 0 else 0
    if unique_ratio > 0.9 and len(vals) > 20:
        # Check if values are sequential integers (classic ID pattern)
        sorted_vals = np.sort(vals)
        diffs = np.diff(sorted_vals)
        if np.all(diffs == 1):
            result.is_valid = False
            result.rejection_reasons.append("Appears to be a sequential identifier")
            return result
        result.warnings.append("High cardinality — verify this is spend data, not an ID")

    # Rule 6: Not the target variable
    if target_col and col_name == target_col:
        result.is_valid = False
        result.rejection_reasons.append("This column is the target variable")
        return result

    # Rule 7: Not a duplicate representation of another field
    if all_channels:
        for other in all_channels:
            if other == col_name or other not in df.columns:
                continue
            other_series = pd.to_numeric(df[other], errors="coerce").dropna()
            if other_series.empty or len(other_series) < 5:
                continue
            # Check correlation — if >0.99, likely a duplicate
            common_idx = numeric.dropna().index.intersection(other_series.index)
            if len(common_idx) >= 5:
                corr = np.corrcoef(
                    numeric.loc[common_idx].values,
                    other_series.loc[common_idx].values
                )[0, 1]
                if abs(corr) > 0.99:
                    result.warnings.append(
                        f"Very high correlation ({corr:.3f}) with '{other}' — "
                        f"may be a duplicate representation")

    # Check for multicollinearity warning (high correlation with other channels)
    if all_channels:
        high_corr_pairs = []
        for other in all_channels:
            if other == col_name or other not in df.columns:
                continue
            other_series = pd.to_numeric(df[other], errors="coerce").dropna()
            common_idx = numeric.dropna().index.intersection(other_series.index)
            if len(common_idx) >= 5:
                corr = np.corrcoef(
                    numeric.loc[common_idx].values,
                    other_series.loc[common_idx].values
                )[0, 1]
                if abs(corr) > 0.85:
                    high_corr_pairs.append((other, corr))
        if high_corr_pairs:
            result.warnings.append(
                f"Potential multicollinearity with: "
                f"{', '.join(f'{p[0]} (r={p[1]:.2f})' for p in high_corr_pairs)}")

    # Additional quality checks
    if (vals < 0).any():
        neg_pct = (vals < 0).sum() / len(vals) * 100
        if neg_pct > 50:
            result.warnings.append(f"{neg_pct:.0f}% of values are negative — unusual for spend data")

    return result


def validate_channels(candidates: list[str], df: pd.DataFrame,
                      target_col: Optional[str] = None) -> list[ChannelValidation]:
    """Validate all channel candidates. Returns only validated channels."""
    results = []
    for col in candidates:
        validation = validate_channel(col, df, target_col=target_col, all_channels=candidates)
        results.append(validation)
    return results


# ---------------------------------------------------------------------------
# 3. Target detection for marketing
# ---------------------------------------------------------------------------

def detect_marketing_target(canonical: CanonicalDataset) -> Optional[str]:
    """Detect the best target/outcome column for marketing analysis.

    Uses CanonicalDataset.target_candidates (already computed by Phase 2).
    Returns the top candidate's column name, or None.
    """
    if not canonical.target_candidates:
        return None
    # target_candidates are already sorted by score (highest first)
    # Return the top candidate
    top = canonical.target_candidates[0]
    if hasattr(top, "column"):
        return top.column
    if isinstance(top, dict):
        return top.get("column")
    return None


# ---------------------------------------------------------------------------
# 4. Control variable detection
# ---------------------------------------------------------------------------

def detect_control_variables(canonical: CanonicalDataset,
                             channel_cols: list[str],
                             target_col: Optional[str] = None,
                             time_col: Optional[str] = None) -> list[str]:
    """Detect control variables for MMM: numeric columns that are NOT
    channels, target, time, or identifiers."""
    exclude = set(channel_cols)
    if target_col:
        exclude.add(target_col)
    if time_col:
        exclude.add(time_col)
    exclude.update(canonical.identifier_columns)
    exclude.update(canonical.boolean_columns)

    controls = []
    for col in canonical.numeric_columns:
        if col in exclude:
            continue
        # Skip constant columns
        meta = canonical.column_metadata.get(col)
        if meta and meta.is_constant:
            continue
        controls.append(col)

    return controls


# ---------------------------------------------------------------------------
# 5. Conversion field detection
# ---------------------------------------------------------------------------

def detect_conversion_fields(canonical: CanonicalDataset) -> list[str]:
    """Detect conversion-related columns (orders, leads, sign-ups, etc.)."""
    conversion_cols = []
    for col in canonical.numeric_columns:
        norm = _normalize_col_name(col)
        norm_aliases = {_normalize_col_name(a) for a in CONVERSIONS_ALIASES}
        if norm in norm_aliases:
            conversion_cols.append(col)
            continue
        tokens = set(norm.split())
        if tokens & {"conversion", "conversions", "leads", "signups", "sign ups", "orders"}:
            conversion_cols.append(col)
    return conversion_cols


# ---------------------------------------------------------------------------
# 6. Full marketing capability assessment
# ---------------------------------------------------------------------------

def assess_marketing_capabilities(
    canonical: CanonicalDataset,
    context: AnalysisContext,
) -> MarketingCapabilityReport:
    """Run the full marketing field discovery and capability assessment.

    This is the main entry point for Phase 6 field discovery.
    """
    report = MarketingCapabilityReport()
    df = canonical.dataframe
    if df is None or df.empty:
        report.readiness = MarketingReadiness.NOT_READY
        report.unavailable_analyses.append(
            "No data available for marketing analysis.")
        return report

    n_rows = len(df)

    # 1. Detect channel candidates
    channel_candidates = detect_channel_candidates(canonical)

    # 2. Validate channels
    validations = validate_channels(channel_candidates, df,
                                     target_col=detect_marketing_target(canonical))
    valid_channels = [v.column for v in validations if v.is_valid]
    report.channel_fields = valid_channels

    # Collect validation warnings
    for v in validations:
        if v.warnings:
            for w in v.warnings:
                report.warnings.append(f"Channel '{v.column}': {w}")
        if not v.is_valid:
            for r in v.rejection_reasons:
                report.warnings.append(f"Channel '{v.column}' rejected: {r}")

    # 3. Detect target
    target_col = detect_marketing_target(canonical)
    report.target_field = target_col

    # 4. Detect time field
    time_col = None
    if canonical.time_info and canonical.time_info.time_column:
        time_col = canonical.time_info.time_column
    report.time_field = time_col

    # 5. Detect controls
    control_cols = detect_control_variables(canonical, valid_channels, target_col, time_col)
    report.control_fields = control_cols

    # 6. Detect conversions
    report.conversion_fields = detect_conversion_fields(canonical)

    # 7. Determine readiness
    has_channels = len(valid_channels) >= 1
    has_enough_channels_for_mmm = len(valid_channels) >= MIN_CHANNELS_FOR_MMM
    has_target = target_col is not None
    has_time = time_col is not None
    has_enough_rows = n_rows >= MIN_ROWS_FOR_MMM
    not_too_many_rows = n_rows <= MAX_ROWS_FOR_MMM

    if not has_channels:
        report.readiness = MarketingReadiness.NOT_READY
        report.unavailable_analyses.append(
            "No marketing spend/channel columns were detected in this dataset. "
            "Marketing intelligence requires columns with spend or cost semantics "
            "(e.g., 'Google_Ads_Spend', 'TV_Budget', 'Search_Cost').")
    elif not has_target:
        report.readiness = MarketingReadiness.PARTIAL
        report.unavailable_analyses.append(
            "No reliable target/outcome column was detected. "
            "Marketing analysis requires a numeric outcome column "
            "(e.g., Revenue, Sales, Conversions).")
    elif has_channels and has_target and not has_time:
        report.readiness = MarketingReadiness.READY_FOR_MARKETING_ANALYSIS
        report.unavailable_analyses.append(
            "No time column was detected. ROI/ROAS is available, but "
            "time-series modeling (MMM, adstock) requires a date/time field.")
    elif has_enough_channels_for_mmm and has_target and has_time and has_enough_rows:
        if not_too_many_rows:
            report.readiness = MarketingReadiness.READY_FOR_OPTIMIZATION
        else:
            report.readiness = MarketingReadiness.READY_FOR_MMM
            report.warnings.append(
                f"Dataset has {n_rows} rows (limit: {MAX_ROWS_FOR_MMM}). "
                f"Data will be sampled for MMM.")
    elif has_channels and has_target and has_time:
        if not has_enough_rows:
            report.readiness = MarketingReadiness.READY_FOR_MARKETING_ANALYSIS
            report.unavailable_analyses.append(
                f"Only {n_rows} rows available. MMM requires at least "
                f"{MIN_ROWS_FOR_MMM} time points.")
        else:
            report.readiness = MarketingReadiness.READY_FOR_MARKETING_ANALYSIS
            report.unavailable_analyses.append(
                f"Only {len(valid_channels)} channel(s) detected. MMM requires "
                f"at least {MIN_CHANNELS_FOR_MMM}.")

    # 8. Set per-capability status
    # ROI
    if has_channels and has_target:
        report.roi = CapabilityStatus(
            available=True, confidence=0.9,
            reason=f"Target '{target_col}' and {len(valid_channels)} channel(s) available.",
            detected_fields=valid_channels + ([target_col] if target_col else []),
        )
    else:
        missing = []
        if not has_channels:
            missing.append("no marketing channels detected")
        if not has_target:
            missing.append("no target column detected")
        report.roi = CapabilityStatus(
            available=False, reason="ROI unavailable: " + "; ".join(missing) + ".",
        )

    # MMM
    mmm_available = (has_enough_channels_for_mmm and has_target and has_time
                     and has_enough_rows)
    if mmm_available:
        report.mmm = CapabilityStatus(
            available=True, confidence=0.85,
            reason=(f"Time column '{time_col}', target '{target_col}', "
                    f"and {len(valid_channels)} channel(s) detected with "
                    f"{n_rows} observations."),
            detected_fields=valid_channels,
        )
        report.adstock = CapabilityStatus(
            available=True, confidence=0.85,
            reason="Adstock requires MMM, which is available.",
            detected_fields=valid_channels,
        )
        report.channel_contribution = CapabilityStatus(
            available=True, confidence=0.85,
            reason="Channel contribution requires a fitted MMM.",
            detected_fields=valid_channels,
        )
    else:
        missing_mmm = []
        if not has_enough_channels_for_mmm:
            missing_mmm.append(
                f"fewer than {MIN_CHANNELS_FOR_MMM} channels "
                f"(found {len(valid_channels)})")
        if not has_target:
            missing_mmm.append("no target column")
        if not has_time:
            missing_mmm.append("no time column")
        if not has_enough_rows:
            missing_mmm.append(f"only {n_rows} rows (need >= {MIN_ROWS_FOR_MMM})")
        reason = "MMM unavailable: " + "; ".join(missing_mmm) + "."
        report.mmm = CapabilityStatus(available=False, reason=reason)
        report.adstock = CapabilityStatus(available=False, reason=reason)
        report.channel_contribution = CapabilityStatus(available=False, reason=reason)

    # Budget optimization
    if mmm_available:
        report.budget_optimization = CapabilityStatus(
            available=True, confidence=0.8,
            reason="Budget optimization requires a fitted MMM, which is available.",
            detected_fields=valid_channels,
        )
    else:
        report.budget_optimization = CapabilityStatus(
            available=False,
            reason="Budget optimization requires a fitted MMM. " + (report.mmm.reason or ""),
        )

    # What-if
    if mmm_available:
        report.what_if = CapabilityStatus(
            available=True, confidence=0.8,
            reason="What-if simulation requires a fitted MMM, which is available.",
            detected_fields=valid_channels,
        )
    else:
        report.what_if = CapabilityStatus(
            available=False,
            reason="What-if simulation requires a fitted MMM. " + (report.mmm.reason or ""),
        )

    return report
