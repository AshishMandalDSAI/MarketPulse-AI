"""Column-level date parsing with an explicit day/month order decision.

Why this exists: ``pd.to_datetime`` infers ONE format from the first value. For a
column of day-first dates (``17/01/2022`` — the norm in India and most of the
world) whose first value is ambiguous (``03/01/2022``) pandas picks month-first,
then silently returns NaT for every day > 12 and *wrong* dates for the rest
(03/01 -> 1 March). Measured on 104 weekly dd/mm/yyyy dates: 64 NaT, 40 wrong.

``parse_date_column`` looks at the whole column:

* ISO / unambiguous text  -> parsed normally.
* numeric d/m/y strings   -> if any first field > 12 the column is day-first; if
  any second field > 12 it is month-first; if both orders are possible for every
  value the column is *ambiguous*: it is parsed month-first (the pandas
  convention) and the ambiguity is reported so the UI can say so — never silent.
* both > 12 in different rows -> inconsistent; parsed with per-value inference
  and reported as such.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import pandas as pd

_DMY = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})(?:[ T].*)?\s*$")


@dataclass
class DateParseInfo:
    order: str = "iso_or_text"          # dayfirst | monthfirst | ambiguous | inconsistent | iso_or_text | datetime
    parsed: int = 0
    unparseable: int = 0
    note: str = ""
    examples_unparseable: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(order=self.order, parsed=self.parsed, unparseable=self.unparseable, note=self.note,
                    examples_unparseable=self.examples_unparseable)


def detect_date_order(series: pd.Series) -> str:
    """Returns dayfirst / monthfirst / ambiguous / inconsistent, or iso_or_text when
    the values are not numeric d/m/y strings."""
    s = series.dropna().astype(str)
    s = s[s.str.strip() != ""]
    if s.empty:
        return "iso_or_text"
    m = s.str.extract(_DMY)
    m = m.dropna()
    if len(m) < 0.8 * len(s):          # mostly not d/m/y strings
        return "iso_or_text"
    a, b = m[0].astype(int), m[1].astype(int)
    first_gt12, second_gt12 = bool((a > 12).any()), bool((b > 12).any())
    if first_gt12 and second_gt12:
        return "inconsistent"
    if first_gt12:
        return "dayfirst"
    if second_gt12:
        return "monthfirst"
    return "ambiguous"


def parse_date_column(series: pd.Series) -> tuple[pd.Series, DateParseInfo]:
    """Parses a column into datetime64 using one consistent day/month order."""
    info = DateParseInfo()
    if pd.api.types.is_datetime64_any_dtype(series):
        info.order = "datetime"
        info.parsed = int(series.notna().sum())
        return series, info
    if pd.api.types.is_numeric_dtype(series):
        # Plain numbers are not dates here (they would parse as epoch offsets).
        out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        info.unparseable = int(series.notna().sum())
        info.note = "Numeric values are not interpreted as dates."
        return out, info

    raw = series.astype("string")
    blank = raw.isna() | (raw.str.strip() == "")
    order = detect_date_order(series)
    info.order = order
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if order == "dayfirst":
            out = pd.to_datetime(raw, errors="coerce", dayfirst=True, format="mixed")
        elif order in ("monthfirst", "ambiguous"):
            out = pd.to_datetime(raw, errors="coerce", dayfirst=False, format="mixed")
        elif order == "inconsistent":
            out = pd.to_datetime(raw, errors="coerce", format="mixed")
        else:
            out = pd.to_datetime(raw, errors="coerce")
            if out[~blank].isna().mean() > 0.2:   # non-uniform text formats
                out = pd.to_datetime(raw, errors="coerce", format="mixed")
    out = pd.Series(out, index=series.index)
    bad = out.isna() & ~blank
    info.parsed = int(out.notna().sum())
    info.unparseable = int(bad.sum())
    info.examples_unparseable = [str(v) for v in series[bad].head(3).tolist()]
    if order == "dayfirst":
        info.note = "Dates read as day/month/year (a day value above 12 was found)."
    elif order == "monthfirst":
        info.note = "Dates read as month/day/year (a day value above 12 was found in the second position)."
    elif order == "ambiguous":
        info.note = ("Date order is ambiguous (no value has a day above 12); dates were read as "
                     "month/day/year. If your dates are day/month/year, check the date range shown.")
    elif order == "inconsistent":
        info.note = ("Dates mix day/month and month/day order; each value was interpreted on its own. "
                     "Check the source data.")
    return out, info
