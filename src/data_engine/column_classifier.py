"""Semantic role detection for arbitrary dataset columns.

Strategy, in order:
  (a) exact alias match            -> confidence ~0.95
  (b) fuzzy/substring alias match  -> confidence ~0.70-0.80 (length-guarded)
  (c) statistical/dtype fallback   -> confidence ~0.40-0.55

LOW-confidence guesses are never auto-applied by downstream code — that is
enforced in data_standardizer.py / ColumnMapping.clear_unconfirmed_low_confidence.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_engine.role_lexicon import is_trade_or_promotion_spend_name
from config.column_aliases import (
    ALIAS_REGISTRY, SPEND_ALIASES, SPEND_GENERIC_ALIASES, CHANNEL_SPEND_ALIASES,
)
from src.data_engine.schemas import ColumnMapping, FieldGuess
from src.data_engine.profiler import _infer_column_type

# Minimum normalized-token length guard for fuzzy/substring matching, so
# short alias fragments like "ig" don't match inside unrelated words like
# "digital" or "config".
MIN_FUZZY_TOKEN_LEN = 4


def _normalize(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[_\-./]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _exact_match(norm_col: str, aliases: set) -> bool:
    return norm_col in aliases


def _fuzzy_match(norm_col: str, aliases: set) -> bool:
    """Substring match, guarded against short-token false positives."""
    for alias in aliases:
        if len(alias) < MIN_FUZZY_TOKEN_LEN:
            continue
        # word-boundary-ish containment: alias appears as a whole token run
        if alias in norm_col or norm_col in alias:
            return True
    return False


def _best_alias_guess(norm_col: str, aliases: set):
    if _exact_match(norm_col, aliases):
        return 0.95, "alias-exact"
    if _fuzzy_match(norm_col, aliases):
        return 0.75, "alias-fuzzy"
    return None


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Plain numbers (IDs, scores, amounts) parse as epoch nanoseconds in
    # pd.to_datetime, which would make any integer column "look like" a date.
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().head(30)
    if sample.empty:
        return False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.85
    except Exception:
        return False


def _looks_like_currency_measure(series: pd.Series) -> bool:
    """A numeric column with currency-like magnitude/variance: non-negative,
    has meaningful variance, and isn't a near-constant flag/id-like column."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty or len(numeric) < 3:
        return False
    # Infinite values make mean/std NaN, so the variance test below already
    # returned False for such columns; say so explicitly instead of computing
    # NaN statistics (which raised a NumPy RuntimeWarning).
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        return False
    if (numeric < 0).mean() > 0.05:
        return False
    if numeric.nunique() <= 2:
        return False
    if numeric.mean() == 0:
        return False
    cv = numeric.std() / (abs(numeric.mean()) + 1e-9)
    return cv > 0.05


def _detect_channel_spend_columns(columns, normalized):
    """Returns list of FieldGuess for candidate channel/spend columns,
    covering per-brand aliases, generic spend wording, and a statistical
    fallback for numeric columns whose name contains 'spend'/'ads'/'cost'."""
    guesses = []
    used = set()

    for col, norm in zip(columns, normalized):
        # PART 02: trade / promotion expenditure (e.g. Trade_Spend, Promo_Budget) is a business
        # variable, not channel-wise advertising spend. A user can still confirm it as a channel
        # explicitly (channel_columns override), which bypasses this guess.
        if is_trade_or_promotion_spend_name(col):
            continue
        matched = False
        for brand, alias_set in CHANNEL_SPEND_ALIASES.items():
            if norm in alias_set:
                guesses.append(FieldGuess(column=col, confidence=0.93, matched_via="alias-exact"))
                used.add(col)
                matched = True
                break
            if any(len(a) >= MIN_FUZZY_TOKEN_LEN and (a in norm or norm in a) for a in alias_set):
                guesses.append(FieldGuess(column=col, confidence=0.72, matched_via="alias-fuzzy"))
                used.add(col)
                matched = True
                break
        if matched:
            continue
        if norm in SPEND_GENERIC_ALIASES:
            guesses.append(FieldGuess(column=col, confidence=0.85, matched_via="alias-exact"))
            used.add(col)
        elif any(kw in norm for kw in ("spend", "ad cost", "ad budget")) and len(norm) >= MIN_FUZZY_TOKEN_LEN:
            guesses.append(FieldGuess(column=col, confidence=0.68, matched_via="alias-fuzzy"))
            used.add(col)

    return guesses, used


def classify_columns(df: pd.DataFrame) -> ColumnMapping:
    """Produces a ColumnMapping for the given dataframe."""
    columns = list(df.columns)
    normalized = [_normalize(c) for c in columns]

    mapping = ColumnMapping()

    # --- 1. Channel/spend detection (multi-column role) ---------------------
    channel_guesses, channel_used = _detect_channel_spend_columns(columns, normalized)
    mapping.channels = channel_guesses
    mapping.spend_like_columns = [g.column for g in channel_guesses]

    # --- 2. Single-role fields, alias-based ---------------------------------
    single_roles = ["date", "revenue", "sales", "cost", "impressions", "clicks",
                     "conversions", "orders", "profit", "margin"]

    assigned = set(channel_used)

    for role in single_roles:
        aliases = ALIAS_REGISTRY[role]
        best = None  # (col, confidence, matched_via)
        for col, norm in zip(columns, normalized):
            if col in assigned:
                continue
            result = _best_alias_guess(norm, aliases)
            if result is None:
                continue
            confidence, via = result
            if best is None or confidence > best[1]:
                best = (col, confidence, via)
        guess = getattr(mapping, role)
        if best:
            guess.column, guess.confidence, guess.matched_via = best
            assigned.add(best[0])

    # --- 3. Statistical fallback for date & revenue if still unmapped -------
    if not mapping.date.is_mapped:
        candidates = []
        for col, norm in zip(columns, normalized):
            if col in assigned:
                continue
            if _looks_like_datetime(df[col]):
                # earlier columns (closer to the start) score slightly higher
                position_bonus = 0.02 if columns.index(col) < 3 else 0.0
                candidates.append((col, 0.50 + position_bonus))
        if candidates:
            col, conf = max(candidates, key=lambda x: x[1])
            mapping.date.column, mapping.date.confidence, mapping.date.matched_via = col, conf, "statistical"
            assigned.add(col)

    if not mapping.revenue.is_mapped:
        candidates = []
        for col, norm in zip(columns, normalized):
            if col in assigned:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and _looks_like_currency_measure(df[col]):
                # Never offer an identifier-like column (e.g. a 0..N row/customer
                # ID) as the revenue guess — same rule the profiler uses.
                if _infer_column_type(df[col], len(df)) == "identifier":
                    continue
                candidates.append((col, 0.45))
        if candidates:
            col, conf = candidates[0]
            mapping.revenue.column, mapping.revenue.confidence, mapping.revenue.matched_via = col, conf, "statistical"
            assigned.add(col)

    # --- 4. Whatever is left is unmapped -------------------------------------
    mapping.unmapped_columns = [c for c in columns if c not in assigned]

    return mapping
