"""Clarification B (P0, before the split): exact-duplicate removal and conflicting-label reporting.

Independent of leakage screening (no import either way). Conflicting-label rows are NEVER deleted.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MAX_CONFLICT_ROWS_LISTED = 200


def handle_duplicates(df: pd.DataFrame, target: str, features: list, time_col: Optional[str] = None) -> tuple:
    """Return (deduplicated frame, report dict, conflicts DataFrame).

    Exact duplicate = identical on target + feature columns (+ time column in chronological mode); the first
    occurrence (earliest if the frame is already time-sorted) is kept. Conflicting-label case = identical
    feature columns (time ignored) but different targets: all rows retained, counted and reported.
    """
    key_cols = [target] + list(features) + ([time_col] if time_col else [])
    key_cols = list(dict.fromkeys(key_cols))
    dup_mask = df.duplicated(subset=key_cols, keep="first")
    n_dup = int(dup_mask.sum())
    dup_groups = int(df[df.duplicated(subset=key_cols, keep=False)].drop_duplicates(subset=key_cols).shape[0]) \
        if n_dup else 0
    out = df.loc[~dup_mask]

    report = {"exact_duplicate_rows_removed": n_dup, "exact_duplicate_groups": dup_groups,
              "conflicting_label_groups": 0, "conflicting_label_rows": 0, "conflicting_label_share": 0.0}
    conflicts = pd.DataFrame(columns=["group", "rows", "distinct_targets", "source_rows"])
    if features and len(out) > 1:
        work = out[list(features) + [target]].copy()
        work["__row"] = out.index.to_numpy()
        gkey = work.groupby(list(features), dropna=False, sort=False).ngroup()
        work["__g"] = gkey.to_numpy()
        stats = work.groupby("__g")[target].agg(lambda s: s.nunique(dropna=False))
        bad = stats[stats > 1].index
        if len(bad):
            sub = work[work["__g"].isin(bad)]
            report["conflicting_label_groups"] = int(len(bad))
            report["conflicting_label_rows"] = int(len(sub))
            report["conflicting_label_share"] = float(len(sub) / max(len(out), 1))
            rows = []
            for g, part in list(sub.groupby("__g", sort=True))[:MAX_CONFLICT_ROWS_LISTED]:
                rows.append({"group": int(g), "rows": int(len(part)),
                             "distinct_targets": int(part[target].nunique(dropna=False)),
                             "source_rows": ", ".join(str(int(r)) for r in part["__row"].tolist()[:20])})
            conflicts = pd.DataFrame(rows)
    return out, report, conflicts
