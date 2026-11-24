"""Phase 8 Part 4 - Module 17 Prescriptive Analytics.

Thin layer over the existing marketing engines: ``fit_mmm`` (response model), ``_predict_for_allocation`` and
``_get_recent_carry_in`` (the optimiser's own objective), ``simulate_custom_allocation`` / ``simulate_total_budget``
(what-if) and the ROI helpers. The only addition is per-channel absolute bounds (``optimize_budget`` supports only
uniform percentage bounds); with uniform bounds the result equals ``optimize_budget`` (tested)."""
