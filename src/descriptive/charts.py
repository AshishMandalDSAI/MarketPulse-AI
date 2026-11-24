"""Plotly figure builders for Descriptive Statistics (Phase 8 Part 2). No Streamlit.

Every builder returns a ``ChartResult``. A chart is never drawn when it would be
meaningless; instead ``available`` is False and ``message`` explains why in plain
English. Large inputs are sampled deterministically for DRAWING ONLY (the
statistics always use the full data) and the sampling is disclosed in ``notes``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats as _scipy_stats

MAX_POINTS = 20_000          # drawn points for box / violin / density / scatter
MAX_GROUPS = 20              # categories allowed in a grouped chart
MAX_BARS = 30                # bars shown in a bar chart
MIN_BINS, MAX_BINS = 2, 200
CHART_TYPES = ("Histogram", "Box plot", "Violin plot", "Density plot", "Correlation heatmap",
               "Scatter plot", "Bar chart", "Frequency distribution")


@dataclass
class ChartResult:
    available: bool
    figure: Optional[go.Figure] = None
    message: str = ""
    notes: list = field(default_factory=list)
    n_total: int = 0
    n_drawn: int = 0

    @property
    def sampled(self) -> bool:
        return self.available and self.n_drawn < self.n_total


def _unavailable(message: str, n_total: int = 0) -> ChartResult:
    return ChartResult(False, None, message, [], n_total, 0)


def _numeric(series: pd.Series) -> np.ndarray:
    """Finite float values of a series (text numbers are converted only if all of them are numbers)."""
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() != series.notna().sum():
            return np.array([], dtype=float)
        series = converted
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return arr[np.isfinite(arr)]


def _sample_note(n_total: int, n_drawn: int) -> list:
    if n_drawn < n_total:
        return [f"Chart drawn from a fixed random sample of {n_drawn:,} of {n_total:,} values "
                "(seed 0) to keep it fast. The statistics tables use all values."]
    return []


def _sample_frame(df: pd.DataFrame, limit: int = MAX_POINTS) -> pd.DataFrame:
    return df if len(df) <= limit else df.sample(n=limit, random_state=0).sort_index()


def _check_column(df: pd.DataFrame, column) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "There is no data to chart."
    if column is None or column not in df.columns:
        return f"The column {column!r} is not in the active dataset."
    return None


def _numeric_precheck(df, column, min_values=2) -> tuple[Optional[np.ndarray], Optional[ChartResult]]:
    err = _check_column(df, column)
    if err:
        return None, _unavailable(err)
    vals = _numeric(df[column])
    if len(vals) == 0:
        return None, _unavailable(f"'{column}' has no valid numbers to chart (all values are missing, "
                                  "infinite or not numeric).", len(df))
    if len(vals) < min_values:
        return None, _unavailable(f"'{column}' has only {len(vals)} valid value(s); at least {min_values} are needed.",
                                  len(vals))
    if np.ptp(vals) == 0:
        return None, _unavailable(f"Every value of '{column}' is {vals[0]:g}, so this chart would show nothing "
                                  "useful. Pick a column whose values vary.", len(vals))
    return vals, None


def _group_check(df, group_col, max_groups=MAX_GROUPS) -> Optional[str]:
    if group_col is None:
        return None
    if group_col not in df.columns:
        return f"The grouping column {group_col!r} is not in the active dataset."
    n = df[group_col].nunique(dropna=True)
    if n < 1:
        return f"'{group_col}' has no values to group by."
    if n > max_groups:
        return (f"'{group_col}' has {n:,} different values, more than the {max_groups} this chart can show "
                "clearly. Choose a column with fewer categories.")
    return None


def histogram(df: pd.DataFrame, column: str, bins: int = 20) -> ChartResult:
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or not (MIN_BINS <= bins <= MAX_BINS):
        return _unavailable(f"The number of bins must be a whole number from {MIN_BINS} to {MAX_BINS}.")
    vals, bad = _numeric_precheck(df, column)
    if bad:
        return bad
    counts, edges = np.histogram(vals, bins=int(bins))            # full data, no sampling needed
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    fig = go.Figure(go.Bar(x=centers, y=counts, width=widths, customdata=np.stack([edges[:-1], edges[1:]], axis=1),
                           hovertemplate="From %{customdata[0]:.4g} to %{customdata[1]:.4g}<br>Count: %{y}<extra></extra>",
                           name="Count"))
    fig.update_layout(title=f"Histogram of {column}", xaxis_title=column, yaxis_title="Number of rows", bargap=0.02)
    return ChartResult(True, fig, "", [f"{len(vals):,} valid values in {int(bins)} equal-width bins."],
                       len(vals), len(vals))


def _grouped_values(df, value_col, group_col, limit=MAX_POINTS):
    sub = df[[value_col] + ([group_col] if group_col else [])].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub[np.isfinite(sub[value_col].to_numpy(dtype=float, na_value=np.nan))]
    if group_col:
        sub = sub[sub[group_col].notna()]
    n_total = len(sub)
    return _sample_frame(sub, limit), n_total


def _box_or_violin(df, column, group_col, kind: str) -> ChartResult:
    vals, bad = _numeric_precheck(df, column)
    if bad:
        return bad
    gerr = _group_check(df, group_col)
    if gerr:
        return _unavailable(gerr)
    sub, n_total = _grouped_values(df, column, group_col)
    cls = go.Box if kind == "box" else go.Violin
    extra = dict(boxpoints="outliers") if kind == "box" else dict(box_visible=True, meanline_visible=True)
    fig = go.Figure()
    if group_col:
        for g, part in sub.groupby(group_col, sort=True, observed=True):
            fig.add_trace(cls(y=part[column], name=str(g), **extra))
    else:
        fig.add_trace(cls(y=sub[column], name=column, **extra))
    title = ("Box plot" if kind == "box" else "Violin plot") + f" of {column}" + (f" by {group_col}" if group_col else "")
    fig.update_layout(title=title, yaxis_title=column, xaxis_title=group_col or "", showlegend=False)
    return ChartResult(True, fig, "", _sample_note(n_total, len(sub)), n_total, len(sub))


def box_plot(df, column, group_col=None) -> ChartResult:
    return _box_or_violin(df, column, group_col, "box")


def violin_plot(df, column, group_col=None) -> ChartResult:
    return _box_or_violin(df, column, group_col, "violin")


def _kde_curve(values: np.ndarray, points: int = 200):
    kde = _scipy_stats.gaussian_kde(values)
    lo, hi = values.min(), values.max()
    pad = (hi - lo) * 0.05
    xs = np.linspace(lo - pad, hi + pad, points)
    return xs, kde(xs)


def density_plot(df, column, group_col=None) -> ChartResult:
    vals, bad = _numeric_precheck(df, column, min_values=3)
    if bad:
        return bad
    gerr = _group_check(df, group_col, max_groups=10)
    if gerr:
        return _unavailable(gerr)
    sub, n_total = _grouped_values(df, column, group_col)
    fig = go.Figure()
    skipped = []
    try:
        if group_col:
            for g, part in sub.groupby(group_col, sort=True, observed=True):
                v = part[column].to_numpy(dtype=float)
                if len(v) < 3 or np.ptp(v) == 0:
                    skipped.append(str(g))
                    continue
                xs, ys = _kde_curve(v)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=str(g)))
        else:
            xs, ys = _kde_curve(sub[column].to_numpy(dtype=float))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy", name=column))
    except (np.linalg.LinAlgError, ValueError):
        return _unavailable(f"A density curve cannot be estimated for '{column}' (the values are too similar).")
    if not fig.data:
        return _unavailable("No group has enough varying values (at least 3) for a density curve.")
    notes = _sample_note(n_total, len(sub))
    if skipped:
        notes.append("Groups left out (fewer than 3 values or no variation): " + ", ".join(skipped) + ".")
    fig.update_layout(title=f"Density of {column}" + (f" by {group_col}" if group_col else ""),
                      xaxis_title=column, yaxis_title="Estimated density")
    return ChartResult(True, fig, "", notes + ["Smooth estimate (Gaussian kernel); it is not a count."],
                       n_total, len(sub))


def correlation_heatmap(matrix, title: str = "") -> ChartResult:
    """``matrix``: a PairwiseMatrix of kind 'correlation'."""
    if matrix is None or len(getattr(matrix, "columns", [])) < 2:
        return _unavailable("Correlation needs at least two numeric columns that are not identifiers.")
    z = np.array([[np.nan if v is None else v for v in row] for row in matrix.values], dtype=float)
    off = z[~np.eye(len(z), dtype=bool)]
    if np.isnan(off).all():
        return _unavailable("Every pair of columns has an undefined correlation (for example constant columns "
                            "or too few shared rows), so there is nothing to show.")
    text = np.where(np.isnan(z), "", np.char.mod("%.2f", np.nan_to_num(z)))
    fig = go.Figure(go.Heatmap(
        z=z, x=matrix.columns, y=matrix.columns, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
        text=text, texttemplate="%{text}" if len(z) <= 15 else None,
        hovertemplate="%{y} vs %{x}<br>correlation: %{z:.3f}<extra></extra>", colorbar=dict(title="r"),
        hoverongaps=False))
    fig.update_layout(title=title or f"Correlation heatmap ({matrix.method})",
                      yaxis=dict(autorange="reversed"))
    notes = ["Blank cells are undefined (constant column or too few shared rows) - they are not zero.",
             "Correlation does not show that one thing causes another."]
    return ChartResult(True, fig, "", notes, len(z), len(z))


def scatter_plot(df, x: str, y: str, color: Optional[str] = None) -> ChartResult:
    for c in (x, y):
        err = _check_column(df, c)
        if err:
            return _unavailable(err)
    if x == y:
        return _unavailable("Choose two different columns for a scatter plot.")
    gerr = _group_check(df, color, max_groups=10)
    if gerr:
        return _unavailable(gerr)
    sub = pd.DataFrame({"x": pd.to_numeric(df[x], errors="coerce"), "y": pd.to_numeric(df[y], errors="coerce")})
    if color:
        sub["g"] = df[color]
    sub = sub[np.isfinite(sub["x"].to_numpy(dtype=float, na_value=np.nan))
              & np.isfinite(sub["y"].to_numpy(dtype=float, na_value=np.nan))]
    if color:
        sub = sub[sub["g"].notna()]
    n_total = len(sub)
    if n_total < 2:
        return _unavailable(f"Only {n_total} row(s) have valid values in both '{x}' and '{y}'; at least 2 are needed.",
                            n_total)
    if sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
        flat = x if sub["x"].nunique() < 2 else y
        return _unavailable(f"'{flat}' has the same value in every plotted row, so a scatter plot would show a flat "
                            "line. Pick columns whose values vary.", n_total)
    drawn = _sample_frame(sub)
    fig = go.Figure()
    if color:
        for g, part in drawn.groupby("g", sort=True, observed=True):
            fig.add_trace(go.Scatter(x=part["x"], y=part["y"], mode="markers", name=str(g)))
    else:
        fig.add_trace(go.Scatter(x=drawn["x"], y=drawn["y"], mode="markers", name="rows"))
    fig.update_layout(title=f"{y} vs {x}", xaxis_title=x, yaxis_title=y, showlegend=bool(color))
    notes = _sample_note(n_total, len(drawn)) + ["A pattern between two columns does not show that one causes the other."]
    return ChartResult(True, fig, "", notes, n_total, len(drawn))


def frequency_chart(df, column: str, top: int = MAX_BARS) -> ChartResult:
    err = _check_column(df, column)
    if err:
        return _unavailable(err)
    s = df[column]
    s = s[s.notna()]
    if s.map(lambda v: isinstance(v, str) and not v.strip()).any():
        s = s[~s.map(lambda v: isinstance(v, str) and not v.strip())]
    if s.empty:
        return _unavailable(f"'{column}' has no values to count.")
    s = s.map(lambda v: str(v) if isinstance(v, (list, dict, set, tuple)) else v)
    vc = s.astype(str).value_counts()
    vc = vc.sort_index(kind="stable").sort_values(ascending=False, kind="stable")
    n_cat = len(vc)
    shown = vc.iloc[:top]
    pct = shown / len(s) * 100
    fig = go.Figure(go.Bar(x=shown.index.tolist(), y=shown.to_numpy(), customdata=pct.to_numpy(),
                           hovertemplate="%{x}<br>Count: %{y}<br>%{customdata:.1f}% of values<extra></extra>"))
    fig.update_layout(title=f"Frequency of {column}", xaxis_title=column, yaxis_title="Number of rows")
    notes = []
    if n_cat > top:
        notes.append(f"Showing the {top} most frequent of {n_cat:,} categories; the rest are not drawn.")
    return ChartResult(True, fig, "", notes, int(len(s)), int(shown.sum()))


def bar_chart(df, category: str, value: Optional[str] = None, agg: str = "mean", top: int = MAX_BARS) -> ChartResult:
    """Bars of count, or of mean/median/sum of ``value`` per category."""
    if agg not in ("count", "mean", "median", "sum"):
        return _unavailable("The summary must be one of: count, mean, median, sum.")
    if agg == "count" or value is None:
        return frequency_chart(df, category, top)
    for c in (category, value):
        err = _check_column(df, c)
        if err:
            return _unavailable(err)
    if category == value:
        return _unavailable("Choose different columns for the categories and the values.")
    vals = pd.to_numeric(df[value], errors="coerce")
    ok = np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan)) & df[category].notna().to_numpy()
    if int(ok.sum()) == 0:
        return _unavailable(f"No row has both a category in '{category}' and a valid number in '{value}'.")
    n_cat = df.loc[ok, category].nunique()
    if n_cat > top * 4:
        return _unavailable(f"'{category}' has {n_cat:,} different categories - too many for a bar chart. "
                            "Choose a column with fewer categories.")
    sub = pd.DataFrame({"c": df.loc[ok, category].astype(str), "v": vals[ok]})
    res = sub.groupby("c")["v"].agg(agg)
    res = res.sort_values(ascending=False, kind="stable")
    shown = res.iloc[:top]
    fig = go.Figure(go.Bar(x=shown.index.tolist(), y=shown.to_numpy(),
                           hovertemplate="%{x}<br>" + agg + ": %{y:.4g}<extra></extra>"))
    fig.update_layout(title=f"{agg.capitalize()} of {value} by {category}", xaxis_title=category,
                      yaxis_title=f"{agg} of {value}")
    notes = []
    if len(res) > top:
        notes.append(f"Showing the {top} highest of {len(res):,} categories.")
    return ChartResult(True, fig, "", notes, int(ok.sum()), int(ok.sum()))
