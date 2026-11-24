"""Descriptive Statistics engine (Phase 8 Part 2). Pure pandas/numpy - no Streamlit.

Conventions (also written into ``METHODOLOGY`` and the exports):

* Missing percentage = missing / ROWS IN THE DATASET * 100 (the denominator
  includes the missing rows). Blank text is counted as missing; +/-infinity is
  counted separately and is NOT missing. Statistics use only valid (finite)
  observations. Missing values are never turned into zero.
* Percentiles / quartiles: linear interpolation between closest ranks
  (numpy default, Hyndman-Fan type 7).
* Variance / standard deviation: sample (n-1) and population (n) versions.
* Skewness: adjusted Fisher-Pearson G1 (same as ``scipy.stats.skew(bias=False)``),
  needs >= 3 valid values that are not all identical.
* Kurtosis: EXCESS kurtosis G2 (normal = 0, ``scipy.stats.kurtosis(bias=False)``),
  needs >= 4 valid values that are not all identical.
* Outliers: Tukey fences, value < Q1 - k*IQR or value > Q3 + k*IQR (k = 1.5 by default).
* Correlation: Pearson or Spearman on pairwise complete observations, needs >= 3
  pairs and variation in both columns; otherwise undefined (never 0).
* Covariance: sample (n-1, needs >= 2 pairs) and population (n, needs >= 1 pair).
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from src.common.dataset_summary import summarize_dataset
from src.data_engine.canonical import IDENTIFIER_CARDINALITY_RATIO
from src.descriptive.contracts import (
    BooleanSummary, CategoricalSummary, DatetimeSummary, DescriptiveReport,
    DescriptiveSettings, IdentifierSummary, MissingSummary, NumericSummary,
    OutlierSummary, PairwiseMatrix,
)

MIN_PAIRS_CORRELATION = 3
MIN_PAIRS_COVARIANCE_SAMPLE = 2

METHODOLOGY = {
    "Missing values": "Missing % = missing cells / rows in the dataset x 100. Blank text counts as missing. "
                      "Infinite numbers are reported separately and excluded from statistics. Missing values "
                      "are never replaced by zero.",
    "Percentiles and quartiles": "Linear interpolation between the closest ranks (numpy default, Hyndman-Fan type 7), "
                                 "the same for every column.",
    "Variance and standard deviation": "Sample versions divide by n-1 (need 2+ values); population versions divide by n. "
                                       "With exactly one value the sample versions are undefined; the population "
                                       "variance of a single value is 0.",
    "Skewness": "Adjusted Fisher-Pearson coefficient G1. Needs at least 3 values that are not all identical; "
                "otherwise it is reported as undefined, not as 0.",
    "Kurtosis": "Excess kurtosis G2 (a normal distribution gives 0). Needs at least 4 values that are not all "
                "identical; otherwise it is undefined, not 0.",
    "Mode": "All values that occur most often, listed in ascending order. If every value occurs once there is "
            "no mode. Ties are shown in full.",
    "Outliers": "Tukey fences: a value is flagged if it is below Q1 - k x IQR or above Q3 + k x IQR (k = 1.5 by "
                "default). A flag is a prompt to look closer, NOT proof that a value is wrong. No record is "
                "removed or changed.",
    "Correlation": "Pearson (straight-line relationship) or Spearman (rank-based) on rows where both columns have "
                   "values. Needs at least 3 such rows and some variation in both columns, otherwise it is "
                   "undefined. Correlation does not show that one thing causes another.",
    "Covariance": "Sample covariance divides by n-1 (needs 2+ pairs); population covariance divides by n. Uses "
                  "pairwise complete rows. Its size depends on the units of the columns.",
    "Column types": "Types come from the platform's shared column detector. Text columns whose every value is a "
                    "number are converted for the statistics (and listed under type decisions); a column with "
                    "some numeric-looking and some invalid values is NOT converted. Identifier-like columns "
                    "are described by counts only and left out of correlation by default.",
    "Dates": "Text dates are read with month-first interpretation for ambiguous formats. Values that cannot "
             "be read are counted as invalid, not dropped silently. Time-zone-aware dates are shown in UTC "
             "when zones are mixed.",
}

OUTLIER_DISCLAIMER = ("An outlier flag only means a value is unusual compared with the rest of its column. "
                      "It can be a data error, but it can also be a real and important observation.")

_BOOL_TRUE = {"true", "t", "yes", "y", "1"}
_BOOL_FALSE = {"false", "f", "no", "n", "0"}
_LEADING_ZERO = re.compile(r"^[+-]?0\d+$")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _py(v):
    """numpy scalar -> python scalar."""
    return v.item() if hasattr(v, "item") else v


def _fin(v) -> Optional[float]:
    """float or None when not finite."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _label(value) -> str:
    return value if isinstance(value, str) else str(value)


def _blank_mask(series: pd.Series) -> np.ndarray:
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return series.map(lambda v: isinstance(v, str) and not v.strip()).to_numpy(dtype=bool)
    return np.zeros(len(series), dtype=bool)


def _missing_mask(series: pd.Series) -> tuple[np.ndarray, int]:
    """(missing_like mask, blank count). NaN/None/NaT/NA plus blank text."""
    isna = series.isna().to_numpy(dtype=bool)
    blank = _blank_mask(series) & ~isna
    return isna | blank, int(blank.sum())


def _tame_unhashable(series: pd.Series) -> pd.Series:
    if series.dtype == object and series.map(lambda v: isinstance(v, (list, dict, set, tuple))).any():
        return series.map(lambda v: str(v) if isinstance(v, (list, dict, set, tuple)) else v)
    return series


def _unique_names(columns) -> list:
    seen: dict = {}
    names = []
    for c in columns:
        base = str(c)
        if base in seen:
            seen[base] += 1
            names.append(f"{base} ({seen[base]})")
        else:
            seen[base] = 1
            names.append(base)
    return names


def validate_percentiles(percentiles) -> tuple:
    return DescriptiveSettings(percentiles=tuple(percentiles)).validate().percentiles


# ---------------------------------------------------------------------------
# Numeric statistics
# ---------------------------------------------------------------------------

def _moments(x: np.ndarray):
    """(mean, sum of squared deviations, m2, m3, m4) with exactly-rounded sums, or None on overflow."""
    n = len(x)
    try:
        total = math.fsum(x.tolist())
        mean = total / n
        with np.errstate(all="ignore"):
            dev = x - mean
            ss = math.fsum((dev * dev).tolist())
            m3 = math.fsum((dev ** 3).tolist()) / n
            m4 = math.fsum((dev ** 4).tolist()) / n
    except OverflowError:
        return None
    if not all(math.isfinite(v) for v in (total, mean, ss, m3, m4)):
        return None
    return total, mean, ss, ss / n, m3, m4


def summarize_numeric_values(x: np.ndarray, column: str = "value", *, sample_size: Optional[int] = None,
                             missing_count: int = 0, blank_count: int = 0, infinite_count: int = 0,
                             settings: Optional[DescriptiveSettings] = None, dtype: str = "",
                             converted_from_text: bool = False) -> NumericSummary:
    """Statistics of finite numeric observations ``x`` (1-D float array, no NaN/inf)."""
    settings = (settings or DescriptiveSettings()).validate()
    x = np.asarray(x, dtype=float)
    n = int(len(x))
    size = int(sample_size if sample_size is not None else n + missing_count + infinite_count)
    s = NumericSummary(column=column, dtype=dtype, sample_size=size, valid_count=n,
                       missing_count=int(missing_count), blank_count=int(blank_count),
                       missing_pct=(missing_count / size * 100.0) if size else 0.0,
                       infinite_count=int(infinite_count), converted_from_text=converted_from_text)
    if converted_from_text:
        s.warnings.append("Values were stored as text and converted to numbers (every non-missing value was numeric).")
    if infinite_count:
        s.warnings.append(f"{infinite_count} infinite value(s) were excluded from every statistic.")
    if n == 0:
        why = "no valid numeric values"
        for k in ("mean", "median", "mode", "minimum", "maximum", "range", "sum", "variance", "standard deviation",
                  "quartiles", "skewness", "kurtosis"):
            s.undefined[k] = f"Undefined: {why}."
        return s

    s.zero_count = int((x == 0).sum())
    s.negative_count = int((x < 0).sum())
    s.minimum, s.maximum = float(x.min()), float(x.max())
    s.value_range = float(s.maximum - s.minimum)
    s.is_constant = bool(s.minimum == s.maximum)
    s.median = float(np.median(x))
    s.q2 = s.median

    mom = _moments(x)
    if mom is None:
        s.undefined["mean"] = "Undefined: the numbers are too large to add up without overflow."
        s.undefined["sum"] = s.undefined["mean"]
        s.undefined["variance"] = s.undefined["standard deviation"] = s.undefined["mean"]
        s.undefined["skewness"] = s.undefined["kurtosis"] = s.undefined["mean"]
    else:
        total, mean, ss, m2, m3, m4 = mom
        s.total, s.mean = total, mean
        s.variance_population = m2
        s.std_population = math.sqrt(m2)
        if n >= 2:
            s.variance_sample = ss / (n - 1)
            s.std_sample = math.sqrt(s.variance_sample)
        else:
            s.undefined["variance"] = "Undefined: the sample variance needs at least 2 values (only 1 is available)."
            s.undefined["standard deviation"] = "Undefined: the sample standard deviation needs at least 2 values (only 1 is available)."
        if s.is_constant and n >= 2:
            s.variance_sample = s.std_sample = s.variance_population = s.std_population = 0.0
        if (s.is_constant or m2 == 0) and n >= 2:
            s.undefined["skewness"] = "Undefined: all values are identical, so there is no spread to measure skewness against."
        elif n < 3:
            s.undefined["skewness"] = f"Undefined: skewness needs at least 3 values ({n} available)."
        else:
            s.skewness = _fin(math.sqrt(n * (n - 1)) / (n - 2) * m3 / (m2 ** 1.5))
        if (s.is_constant or m2 == 0) and n >= 2:
            s.undefined["kurtosis"] = "Undefined: all values are identical, so there is no spread to measure kurtosis against."
        elif n < 4:
            s.undefined["kurtosis"] = f"Undefined: kurtosis needs at least 4 values ({n} available)."
        else:
            g2 = m4 / (m2 * m2) - 3.0
            s.kurtosis = _fin((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6.0))
        for stat in ("skewness", "kurtosis"):
            if getattr(s, stat) is None and stat not in s.undefined:
                s.undefined[stat] = "Undefined: the value is not a finite number for this column."

    s.q1, s.q3 = (float(v) for v in np.quantile(x, [0.25, 0.75]))
    s.iqr = s.q3 - s.q1
    s.percentiles = {float(p): float(np.percentile(x, p)) for p in settings.percentiles}

    vals, counts = np.unique(x, return_counts=True)
    top = int(counts.max())
    if n == 1:
        s.modes, s.mode_count, s.mode_note = [float(vals[0])], 1, "Only one value, so it is the mode."
    elif top == 1:
        s.modes, s.mode_count = [], None
        s.mode_note = "No mode: every value occurs only once."
        s.undefined["mode"] = "Undefined: every value occurs only once."
    else:
        s.modes = [float(v) for v in vals[counts == top]]
        s.mode_count = top
        s.mode_note = (f"{len(s.modes)} values tie for most frequent ({top} times each)."
                       if len(s.modes) > 1 else f"Occurs {top} times.")
    return s


# ---------------------------------------------------------------------------
# Categorical / boolean / datetime
# ---------------------------------------------------------------------------

def _freq_table(series: pd.Series, mask_missing: np.ndarray):
    """Deterministic frequency table: count desc, then label asc."""
    valid = series[~mask_missing]
    if valid.empty:
        return []
    vc = valid.value_counts(dropna=True, sort=False)
    rows = [(_label(k), int(c)) for k, c in vc.items()]
    merged: dict = {}
    for lab, c in rows:                       # distinct values with the same text (1 and "1") are shown together
        merged[lab] = merged.get(lab, 0) + c
    return sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))


def summarize_categorical(series: pd.Series, column: str, settings: DescriptiveSettings) -> CategoricalSummary:
    series = _tame_unhashable(series)
    n = len(series)
    miss, blank = _missing_mask(series)
    table = _freq_table(series, miss)
    valid = n - int(miss.sum())
    s = CategoricalSummary(column=column, dtype=str(series.dtype), sample_size=n, valid_count=valid,
                           missing_count=int(miss.sum()), blank_count=blank,
                           missing_pct=(miss.sum() / n * 100.0) if n else 0.0, unique_count=len(table))
    if blank:
        s.warnings.append(f"{blank} blank text value(s) were counted as missing.")
    if not table:
        s.undefined["mode"] = "Undefined: the column has no values."
        s.undefined["frequency"] = "Undefined: the column has no values."
        return s

    def row(lab, c):
        return {"value": lab, "count": c, "pct": c / valid * 100.0}

    top = table[0][1]
    modes = sorted(lab for lab, c in table if c == top)
    s.modes, s.mode_count, s.mode_pct = modes, top, top / valid * 100.0
    if len(table) == 1:
        s.mode_note = "Only one category."
    elif top == 1:
        s.mode_note = "Every category occurs once, so all of them tie for most frequent."
    elif len(modes) > 1:
        s.mode_note = f"{len(modes)} categories tie for most frequent ({top} times each)."
    s.top_categories = [row(l, c) for l, c in table[:settings.top_n]]
    least = sorted(table, key=lambda kv: (kv[1], kv[0]))[:settings.top_n]
    s.least_frequent = [row(l, c) for l, c in least]
    s.frequency = [row(l, c) for l, c in table[:settings.max_frequency_rows]]
    s.frequency_truncated = len(table) > settings.max_frequency_rows
    s.high_cardinality = len(table) > settings.high_cardinality_threshold
    if s.high_cardinality:
        s.warnings.append(f"{len(table)} different categories: only the {settings.max_frequency_rows} most "
                          "frequent are listed in the frequency table.")
    return s


def _to_bool_array(series: pd.Series, miss: np.ndarray):
    """(bool array, unmapped mask) for the non-missing values."""
    out = np.zeros(len(series), dtype=bool)
    unmapped = np.zeros(len(series), dtype=bool)
    for i, (v, m) in enumerate(zip(series.tolist(), miss)):
        if m:
            continue
        if isinstance(v, (bool, np.bool_)):
            out[i] = bool(v)
        elif isinstance(v, (int, float, np.integer, np.floating)) and v in (0, 1):
            out[i] = bool(v)
        elif isinstance(v, str) and v.strip().lower() in _BOOL_TRUE:
            out[i] = True
        elif isinstance(v, str) and v.strip().lower() in _BOOL_FALSE:
            out[i] = False
        else:
            unmapped[i] = True
    return out, unmapped


def summarize_boolean(series: pd.Series, column: str) -> BooleanSummary:
    n = len(series)
    miss, _ = _missing_mask(series)
    arr, unmapped = _to_bool_array(series, miss)
    valid_mask = ~miss & ~unmapped
    valid = int(valid_mask.sum())
    t = int((arr & valid_mask).sum())
    f = valid - t
    s = BooleanSummary(column=column, dtype=str(series.dtype), sample_size=n, valid_count=valid,
                       missing_count=int(miss.sum()), missing_pct=(miss.sum() / n * 100.0) if n else 0.0,
                       true_count=t, false_count=f)
    if valid:
        s.true_pct, s.false_pct = t / valid * 100.0, f / valid * 100.0
    else:
        s.undefined["percentages"] = "Undefined: the column has no yes/no values."
    if unmapped.any():
        s.warnings.append(f"{int(unmapped.sum())} value(s) could not be read as yes/no and were left out.")
    return s


def _parse_datetimes(series: pd.Series, miss: np.ndarray):
    """(datetime Series (NaT where missing/invalid), invalid mask, tz label, notes)."""
    notes: list = []
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series.copy()
        tz = str(series.dt.tz) if getattr(series.dt, "tz", None) is not None else None
        return parsed, np.zeros(len(series), dtype=bool), tz, notes
    text = series.where(~miss)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(text, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
        if parsed.dtype == object:             # mixed time zones
            parsed = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
            notes.append("Dates had different time zones and were converted to UTC.")
    tz = str(parsed.dt.tz) if getattr(parsed.dt, "tz", None) is not None else None
    if tz is not None and not notes:
        offsets = text.dropna().astype(str).str.extract(r"(Z|[+-]\d{2}:?\d{2})\s*$")[0].dropna().unique()
        if len(offsets) > 1:
            notes.append("Dates had different time zones and were converted to UTC.")
    invalid = parsed.isna().to_numpy(dtype=bool) & ~miss
    return parsed, invalid, tz, notes


def _choose_period(span_days: float) -> str:
    if span_days <= 62:
        return "day"
    if span_days <= 365 * 3:
        return "month"
    return "year"


def summarize_datetime(series: pd.Series, column: str, *, parsed=None, invalid=None, tz=None,
                       notes=None) -> DatetimeSummary:
    n = len(series)
    miss, blank = _missing_mask(series)
    if parsed is None:
        parsed, invalid, tz, notes = _parse_datetimes(series, miss)
    valid_mask = parsed.notna().to_numpy(dtype=bool) & ~miss
    valid = parsed[valid_mask]
    s = DatetimeSummary(column=column, dtype=str(series.dtype), sample_size=n, valid_count=int(len(valid)),
                        missing_count=int(miss.sum()), blank_count=blank,
                        missing_pct=(miss.sum() / n * 100.0) if n else 0.0,
                        invalid_count=int(np.sum(invalid)), timezone=tz)
    s.warnings.extend(notes or [])
    if s.invalid_count:
        s.warnings.append(f"{s.invalid_count} value(s) could not be read as a date; they are reported as invalid, "
                          "not dropped silently.")
    if valid.empty:
        s.undefined["range"] = "Undefined: no valid dates."
        return s
    lo, hi = valid.min(), valid.max()
    s.earliest, s.latest = lo.isoformat(), hi.isoformat()
    s.range_days = float((hi - lo).total_seconds() / 86400.0)
    s.unique_count = int(valid.nunique())
    if len(valid) >= 2 and s.range_days > 0:
        naive = valid.dt.tz_localize(None) if getattr(valid.dt, "tz", None) is not None else valid
        s.period = _choose_period(s.range_days)
        freq = {"day": "D", "month": "M", "year": "Y"}[s.period]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            keys = naive.dt.to_period(freq)
        counts = keys.value_counts().sort_index()
        if len(counts) > 120 and s.period != "year":
            s.period = "month" if s.period == "day" else "year"
            keys = naive.dt.to_period("M" if s.period == "month" else "Y")
            counts = keys.value_counts().sort_index()
        s.period_counts = [{"period": str(k), "count": int(c)} for k, c in counts.items()]
    else:
        s.undefined["period_counts"] = "Not meaningful: fewer than 2 distinct dates."
    return s


# ---------------------------------------------------------------------------
# Column typing
# ---------------------------------------------------------------------------

_DATE_LIKE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}:\d{2}|[A-Za-z]{3,9}\.? \d{1,2}|\d{1,2} [A-Za-z]{3,9}")


def _date_candidate(series: pd.Series, miss: np.ndarray) -> tuple[bool, bool]:
    """(pure_digits, date_like): digit-only text is never read as dates; date-like text has separators,
    a time, or a month name in at least 85% of its values."""
    present = series[~miss].astype(str).str.strip()
    if present.empty:
        return False, False
    pure = bool(present.str.fullmatch(r"\d+").all())
    like = float(present.map(lambda t: bool(_DATE_LIKE.search(t))).mean()) >= 0.85
    return pure, like


class _Plan:
    __slots__ = ("kind", "numeric", "parsed", "invalid", "tz", "notes", "missing", "blank", "infinite",
                 "converted")

    def __init__(self, kind, **kw):
        self.kind = kind
        self.numeric = kw.get("numeric")
        self.parsed = kw.get("parsed")
        self.invalid = kw.get("invalid")
        self.tz = kw.get("tz")
        self.notes = kw.get("notes", [])
        self.missing = kw.get("missing")
        self.blank = kw.get("blank", 0)
        self.infinite = kw.get("infinite", 0)
        self.converted = kw.get("converted", False)


def _numeric_from_text(series: pd.Series, miss: np.ndarray):
    """Float array if EVERY non-missing text value is a number, else None. Also the count that did parse."""
    present = series[~miss]
    if present.empty:
        return None, 0
    text = present.astype(str).str.strip()
    if text.map(lambda t: bool(_LEADING_ZERO.match(t))).any():
        return None, 0                      # "007" would lose its zeros: keep as text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        num = pd.to_numeric(text, errors="coerce")
    ok = int(num.notna().sum())
    if ok != len(text):
        return None, ok
    out = np.full(len(series), np.nan)
    out[~miss] = num.to_numpy(dtype=float)
    return out, ok


def _plan_column(series: pd.Series, shared_type: str, n_rows: int, name: str, conversions: list) -> _Plan:
    series = _tame_unhashable(series)
    miss, blank = _missing_mask(series)
    if miss.all():
        return _Plan("empty", missing=miss, blank=blank)

    def is_flag(values: np.ndarray) -> bool:
        u = np.unique(values[np.isfinite(values)])
        return len(u) == 2 and set(u.tolist()) == {0.0, 1.0}

    def identifier_like(values: np.ndarray) -> bool:
        fin = values[np.isfinite(values)]
        if fin.size == 0 or n_rows <= 20:
            return False
        ratio = len(np.unique(fin)) / n_rows
        return bool(ratio > IDENTIFIER_CARDINALITY_RATIO and np.all(fin % 1 == 0))

    if pd.api.types.is_bool_dtype(series.dtype):
        return _Plan("boolean", missing=miss, blank=blank)
    if pd.api.types.is_datetime64_any_dtype(series):
        p, inv, tz, notes = _parse_datetimes(series, miss)
        return _Plan("datetime", parsed=p, invalid=inv, tz=tz, notes=notes, missing=miss, blank=blank)

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_complex_dtype(series):
        with np.errstate(all="ignore"):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        infinite = int(np.isinf(values).sum())
        values = np.where(miss, np.nan, values)
        if shared_type == "identifier" or (identifier_like(values) and shared_type != "boolean"):
            return _Plan("identifier", missing=miss, blank=blank)
        if is_flag(values):
            conversions.append(f"'{name}': numbers 0/1 treated as a yes/no column.")
            return _Plan("boolean", missing=miss, blank=blank)
        return _Plan("numeric", numeric=values, missing=miss, blank=blank, infinite=infinite)

    # text-like columns
    if series.dtype.name == "category":
        series = series.astype(object)
    if shared_type == "boolean":
        _, unmapped = _to_bool_array(series, miss)
        if not unmapped.any():
            return _Plan("boolean", missing=miss, blank=blank)
    values, ok = _numeric_from_text(series, miss)
    if values is not None:
        infinite = int(np.isinf(values).sum())
        if identifier_like(values) and shared_type == "identifier":
            return _Plan("identifier", missing=miss, blank=blank)
        conversions.append(f"'{name}': {ok} text value(s) converted to numbers (all non-missing values were numeric).")
        if is_flag(values):
            conversions.append(f"'{name}': numbers 0/1 treated as a yes/no column.")
            return _Plan("boolean", missing=miss, blank=blank)
        return _Plan("numeric", numeric=values, missing=miss, blank=blank, infinite=infinite, converted=True)
    if ok:
        total = int((~miss).sum())
        conversions.append(f"'{name}': {ok} of {total} values look like numbers but {total - ok} do not; the column "
                           "was NOT converted and is described as text/categories (nothing was discarded).")
    if shared_type == "identifier":
        return _Plan("identifier", missing=miss, blank=blank)
    pure_digits, date_like = _date_candidate(series, miss)
    if not pure_digits and (shared_type == "datetime" or (shared_type in ("categorical", "text") and date_like)):
        p, inv, tz, notes = _parse_datetimes(series, miss)
        present = int((~miss).sum())
        if present and (present - int(inv.sum())) / present >= 0.85:
            if shared_type != "datetime":
                conversions.append(f"'{name}': text read as dates (every value looks like a date or time).")
            return _Plan("datetime", parsed=p, invalid=inv, tz=tz, notes=notes, missing=miss, blank=blank)
    return _Plan("categorical", missing=miss, blank=blank)


# ---------------------------------------------------------------------------
# Missing values and outliers
# ---------------------------------------------------------------------------

def _missing_summary(names, plans, n_rows) -> MissingSummary:
    rows = []
    any_missing = np.zeros(n_rows, dtype=bool)
    for name, plan in zip(names, plans):
        miss = plan.missing
        any_missing |= miss
        mc = int(miss.sum())
        inv = int(np.sum(plan.invalid)) if plan.invalid is not None else 0
        rows.append({
            "column": name, "type": plan.kind, "rows": n_rows, "missing_count": mc,
            "missing_pct": (mc / n_rows * 100.0) if n_rows else 0.0,
            "blank_text": int(plan.blank), "infinite_count": int(plan.infinite), "invalid_dates": inv,
            "valid_count": int(n_rows - mc - plan.infinite - inv),
            "valid_pct": ((n_rows - mc - plan.infinite - inv) / n_rows * 100.0) if n_rows else 0.0,
        })
    total = n_rows * len(names)
    mcells = int(sum(r["missing_count"] for r in rows))
    rows_with = int(any_missing.sum())
    return MissingSummary(
        rows=sorted(rows, key=lambda r: (-r["missing_pct"], names.index(r["column"]))),
        total_cells=total, missing_cells=mcells,
        missing_cell_pct=(mcells / total * 100.0) if total else 0.0,
        complete_rows=n_rows - rows_with,
        complete_row_pct=((n_rows - rows_with) / n_rows * 100.0) if n_rows else 0.0,
        rows_with_missing=rows_with)


def detect_outliers(values: np.ndarray, column: str = "value", multiplier: float = 1.5,
                    row_labels=None, max_examples: int = 20) -> OutlierSummary:
    """Tukey fences on finite ``values``. Original data is never modified."""
    x = np.asarray(values, dtype=float)
    keep = np.isfinite(x)
    xs = x[keep]
    labels = np.asarray(row_labels if row_labels is not None else np.arange(len(x)), dtype=object)[keep]
    o = OutlierSummary(column=column, valid_count=int(len(xs)), multiplier=float(multiplier))
    if len(xs) < 4:
        o.reason = (f"Outlier detection needs at least 4 valid values ({len(xs)} available); "
                    "quartiles of fewer values are not reliable.")
        return o
    o.available = True
    q1, q3 = (float(v) for v in np.quantile(xs, [0.25, 0.75]))
    iqr = q3 - q1
    o.q1, o.q3, o.iqr = q1, q3, iqr
    o.lower_bound, o.upper_bound = q1 - multiplier * iqr, q3 + multiplier * iqr
    low = xs < o.lower_bound
    high = xs > o.upper_bound
    o.n_low, o.n_high = int(low.sum()), int(high.sum())
    o.n_outliers = o.n_low + o.n_high
    o.outlier_pct = o.n_outliers / len(xs) * 100.0
    if iqr == 0:
        o.warnings.append("The middle half of the values are identical (IQR = 0), so any different value is flagged.")
    flagged = np.where(low | high)[0]
    dist = np.abs(xs[flagged] - np.clip(xs[flagged], o.lower_bound, o.upper_bound))
    order = flagged[np.lexsort((np.arange(len(flagged)), -dist))]
    o.examples_truncated = len(order) > max_examples
    o.examples = [{"row": _py(labels[i]), "value": float(xs[i]), "side": "low" if low[i] else "high"}
                  for i in order[:max_examples]]
    return o


# ---------------------------------------------------------------------------
# Correlation and covariance (pairwise complete observations)
# ---------------------------------------------------------------------------

def _rank(v: np.ndarray) -> np.ndarray:
    return _scipy_stats.rankdata(v, method="average")


def _pearson_pair(x: np.ndarray, y: np.ndarray):
    xc, yc = x - x.mean(), y - y.mean()
    sx, sy = float((xc * xc).sum()), float((yc * yc).sum())
    if sx == 0.0 or sy == 0.0:
        return None
    r = float((xc * yc).sum() / math.sqrt(sx * sy))
    return max(-1.0, min(1.0, r))


def pairwise_matrix(data: dict, kind: str, method: str = "pearson") -> PairwiseMatrix:
    """Correlation (``kind='correlation'``) or covariance (``'covariance_sample'`` / ``'covariance_population'``)
    of the columns in ``data`` ({name: float array, NaN/inf = unavailable}) on pairwise complete rows."""
    names = list(data)
    k = len(names)
    arrays = [np.asarray(data[c], dtype=float) for c in names]
    if kind == "correlation":
        label, min_pairs = method, MIN_PAIRS_CORRELATION
    elif kind == "covariance_sample":
        label, min_pairs = "pearson (covariance)", MIN_PAIRS_COVARIANCE_SAMPLE
    else:
        label, min_pairs = "pearson (covariance)", 1
    out = PairwiseMatrix(kind=kind, method=label, columns=names)
    vals = [[None] * k for _ in range(k)]
    npairs = [[0] * k for _ in range(k)]
    finite = [np.isfinite(a) for a in arrays]
    for i in range(k):
        for j in range(i, k):
            ok = finite[i] & finite[j]
            n = int(ok.sum())
            npairs[i][j] = npairs[j][i] = n
            reason = None
            result = None
            if n < min_pairs:
                reason = f"only {n} row(s) have values in both columns; at least {min_pairs} are needed"
            else:
                x, y = arrays[i][ok], arrays[j][ok]
                if kind == "correlation":
                    if method == "spearman":
                        x, y = _rank(x), _rank(y)
                    result = _pearson_pair(x, y)
                    if result is None:
                        const = names[i] if np.ptp(x) == 0 else names[j]
                        reason = f"'{const}' has identical values in the rows used, so correlation is undefined"
                else:
                    xc, yc = x - x.mean(), y - y.mean()
                    with np.errstate(all="ignore"):
                        denom = (n - 1) if kind == "covariance_sample" else n
                        cov = float((xc * yc).sum() / denom)
                    result = _fin(cov)
                    if result is None:
                        reason = "the covariance is not a finite number (values too large)"
            vals[i][j] = vals[j][i] = result
            if reason:
                out.reasons[f"{names[i]} || {names[j]}"] = reason
    out.values, out.n_pairs = vals, npairs
    return out


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------

def analyze_dataframe(df: pd.DataFrame, settings: Optional[DescriptiveSettings] = None) -> DescriptiveReport:
    """Full descriptive report for any DataFrame. Never modifies ``df``."""
    settings = (settings or DescriptiveSettings()).validate()
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("analyze_dataframe needs a pandas DataFrame.")
    n_rows, n_cols = df.shape
    report = DescriptiveReport(n_rows=int(n_rows), n_cols=int(n_cols), settings=settings.to_dict())
    if n_rows == 0 or n_cols == 0:
        report.warnings.append("The dataset has no rows or no columns, so nothing can be described.")
        return report

    names = _unique_names(df.columns)
    work = df.copy(deep=False)
    work.columns = names
    with np.errstate(all="ignore"):          # infinite values are handled explicitly below; keep the shared detector quiet
        shared = summarize_dataset(work)
    shared_type = {c["column"]: c["detected_type"] for c in shared.columns}
    report.quality = {
        "duplicate_rows": shared.duplicate_rows, "duplicate_pct": (shared.duplicate_rows / n_rows * 100.0),
        "constant_columns": list(shared.constant_columns), "empty_columns": list(shared.empty_columns),
        "memory_mb": shared.memory_mb,
        "high_cardinality_columns": [c["column"] for c in shared.columns
                                     if c["detected_type"] in ("categorical", "text", "identifier")
                                     and c["unique_values"] > settings.high_cardinality_threshold],
    }

    plans = []
    for i, name in enumerate(names):
        plans.append(_plan_column(work.iloc[:, i], shared_type.get(name, "categorical"), n_rows, name,
                                  report.conversions))
    report.column_kinds = {n: p.kind for n, p in zip(names, plans)}
    report.missing = _missing_summary(names, plans, n_rows)

    numeric_data: dict = {}
    for i, (name, plan) in enumerate(zip(names, plans)):
        series = work.iloc[:, i]
        if plan.kind == "numeric":
            v = plan.numeric
            finite = np.isfinite(v)
            s = summarize_numeric_values(v[finite], name, sample_size=n_rows, missing_count=int(plan.missing.sum()),
                                         blank_count=plan.blank, infinite_count=plan.infinite,
                                         settings=settings, dtype=str(series.dtype), converted_from_text=plan.converted)
            report.numeric.append(s)
            numeric_data[name] = np.where(finite, v, np.nan)
            o = detect_outliers(v[finite], name, settings.outlier_multiplier,
                                row_labels=df.index.to_numpy()[finite], max_examples=settings.max_outlier_examples)
            report.outliers.append(o)
        elif plan.kind == "categorical":
            report.categorical.append(summarize_categorical(series, name, settings))
        elif plan.kind == "boolean":
            report.boolean.append(summarize_boolean(series, name))
        elif plan.kind == "datetime":
            report.datetime.append(summarize_datetime(series, name, parsed=plan.parsed, invalid=plan.invalid,
                                                      tz=plan.tz, notes=plan.notes))
        elif plan.kind == "identifier":
            miss = plan.missing
            report.identifiers.append(IdentifierSummary(
                column=name, dtype=str(series.dtype), sample_size=n_rows, valid_count=int(n_rows - miss.sum()),
                missing_count=int(miss.sum()), missing_pct=(miss.sum() / n_rows * 100.0),
                unique_count=int(_tame_unhashable(series).nunique(dropna=True))))
        else:
            report.empty_columns.append(name)

    report.warnings.append("Correlation does not show that one thing causes another.")
    if report.identifiers:
        report.warnings.append("Identifier-like columns are excluded from correlation and covariance by default.")

    used = dict(list(numeric_data.items())[:settings.max_correlation_columns])
    skipped = [c for c in numeric_data if c not in used]
    if len(used) >= 2:
        report.correlation = pairwise_matrix(used, "correlation", settings.correlation_method)
        report.covariance_sample = pairwise_matrix(used, "covariance_sample")
        report.covariance_population = pairwise_matrix(used, "covariance_population")
        for m in (report.correlation, report.covariance_sample, report.covariance_population):
            m.excluded = {c: "left out: more numeric columns than the limit" for c in skipped}
            if skipped:
                m.notes.append(f"Only the first {settings.max_correlation_columns} numeric columns are used "
                               f"({len(skipped)} left out).")
    return report
