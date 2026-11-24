"""Core dataclasses shared across the Universal Data Engine."""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.constants import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, MIN_CHANNELS_FOR_MMM


@dataclass
class FieldGuess:
    """A guess about which real column fills a semantic role (e.g. 'date')."""

    column: Optional[str] = None
    confidence: float = 0.0
    matched_via: str = "none"  # "alias-exact" | "alias-fuzzy" | "statistical" | "none"

    @property
    def is_mapped(self) -> bool:
        return self.column is not None

    @property
    def confidence_level(self) -> str:
        if self.confidence >= CONFIDENCE_HIGH:
            return "HIGH"
        if self.confidence >= CONFIDENCE_MEDIUM:
            return "MEDIUM"
        return "LOW"

    @property
    def needs_confirmation(self) -> bool:
        """True when this guess is MEDIUM or LOW confidence and therefore
        must not be auto-applied without explicit user confirmation."""
        return self.is_mapped and self.confidence < CONFIDENCE_HIGH

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "confidence": round(self.confidence, 3),
            "confidence_level": self.confidence_level,
            "matched_via": self.matched_via,
            "needs_confirmation": self.needs_confirmation,
        }


@dataclass
class ColumnMapping:
    """Holds a FieldGuess per semantic role, plus book-keeping about columns
    that were not (yet) assigned any role."""

    date: FieldGuess = field(default_factory=FieldGuess)
    revenue: FieldGuess = field(default_factory=FieldGuess)
    sales: FieldGuess = field(default_factory=FieldGuess)
    cost: FieldGuess = field(default_factory=FieldGuess)
    impressions: FieldGuess = field(default_factory=FieldGuess)
    clicks: FieldGuess = field(default_factory=FieldGuess)
    conversions: FieldGuess = field(default_factory=FieldGuess)
    orders: FieldGuess = field(default_factory=FieldGuess)
    profit: FieldGuess = field(default_factory=FieldGuess)
    margin: FieldGuess = field(default_factory=FieldGuess)

    # channels is a list of (column, FieldGuess) style entries since there
    # can be many spend-like columns.
    channels: list = field(default_factory=list)  # list[FieldGuess]

    unmapped_columns: list = field(default_factory=list)
    spend_like_columns: list = field(default_factory=list)  # candidate channel cols (incl. low-conf)
    freed_columns: list = field(default_factory=list)  # columns cleared by clear_unconfirmed_low_confidence

    def _single_role_guesses(self):
        return {
            "date": self.date, "revenue": self.revenue, "sales": self.sales,
            "cost": self.cost, "impressions": self.impressions, "clicks": self.clicks,
            "conversions": self.conversions, "orders": self.orders,
            "profit": self.profit, "margin": self.margin,
        }

    def clear_unconfirmed_low_confidence(self, confirmed_roles: Optional[set] = None) -> None:
        """Clear any FieldGuess back to unmapped unless its role is in
        confirmed_roles or its confidence is HIGH. Cleared columns are
        recorded in freed_columns so the standardizer can reclassify them
        by dtype instead of silently dropping them."""
        confirmed_roles = confirmed_roles or set()

        for role, guess in self._single_role_guesses().items():
            if not guess.is_mapped:
                continue
            if role in confirmed_roles or guess.confidence >= CONFIDENCE_HIGH:
                continue
            self.freed_columns.append(guess.column)
            guess.column = None
            guess.confidence = 0.0
            guess.matched_via = "none"

        kept_channels = []
        for guess in self.channels:
            if not guess.is_mapped:
                continue
            if "channels" in confirmed_roles or guess.confidence >= CONFIDENCE_HIGH:
                kept_channels.append(guess)
            else:
                self.freed_columns.append(guess.column)
        self.channels = kept_channels

    def to_dict(self) -> dict:
        d = {role: guess.to_dict() for role, guess in self._single_role_guesses().items()}
        d["channels"] = [g.to_dict() for g in self.channels]
        d["unmapped_columns"] = list(self.unmapped_columns)
        d["spend_like_columns"] = list(self.spend_like_columns)
        d["freed_columns"] = list(self.freed_columns)
        return d


@dataclass
class StandardizedDataset:
    """A dataset that has been mapped into semantic roles, without ever
    forcing columns that don't belong (permissive by design)."""

    df: pd.DataFrame
    date_col: Optional[str] = None
    revenue_col: Optional[str] = None
    channels: list = field(default_factory=list)
    control_cols: list = field(default_factory=list)  # generic numeric measures
    dimension_cols: list = field(default_factory=list)  # categorical
    identifier_cols: list = field(default_factory=list)  # near-unique columns
    company_name: Optional[str] = None
    mapping_report: list = field(default_factory=list)

    @property
    def has_date(self) -> bool:
        return bool(self.date_col)

    @property
    def has_revenue(self) -> bool:
        return bool(self.revenue_col)

    @property
    def has_channels(self) -> bool:
        return len(self.channels) > 0

    @property
    def measure_cols(self) -> list:
        """All numeric measure-like columns: revenue + channels + controls."""
        cols = []
        if self.revenue_col:
            cols.append(self.revenue_col)
        cols.extend(self.channels)
        cols.extend(self.control_cols)
        return list(dict.fromkeys(cols))  # de-dup, preserve order

    def date_range(self):
        if not self.has_date or self.date_col not in self.df.columns:
            return None
        series = pd.to_datetime(self.df[self.date_col], errors="coerce").dropna()
        if series.empty:
            return None
        return (series.min(), series.max())

    def is_ready_for_analysis(self):
        """Deliberately permissive: any dataset with >=1 usable column can be
        standardized and explored (generic KPIs, profiling, AI analyst)."""
        return True, []

    def mmm_readiness(self):
        """Checks the specific requirements for Marketing Mix Modeling:
        date + revenue + >= MIN_CHANNELS_FOR_MMM channel/spend columns."""
        missing = []
        if not self.has_date:
            missing.append("no date/time column identified")
        if not self.has_revenue:
            missing.append("no revenue/outcome column identified")
        if len(self.channels) < MIN_CHANNELS_FOR_MMM:
            missing.append(
                f"fewer than {MIN_CHANNELS_FOR_MMM} marketing spend/channel columns identified "
                f"(found {len(self.channels)})"
            )
        return (len(missing) == 0), missing
