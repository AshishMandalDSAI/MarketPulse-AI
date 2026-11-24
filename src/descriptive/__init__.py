"""Descriptive Statistics (Phase 8 Part 2): computation, charts and export. No Streamlit imports here."""

from src.descriptive.contracts import DescriptiveReport, DescriptiveSettings
from src.descriptive.engine import analyze_dataframe

__all__ = ["DescriptiveReport", "DescriptiveSettings", "analyze_dataframe"]
