"""Phase 6 — Marketing Intelligence Orchestrator.

Single entry point that runs the full marketing analysis pipeline:
  1. Field discovery & capability assessment
  2. MMM fitting (if capable)
  3. Channel contribution (if MMM fitted)
  4. ROI/ROAS (if channels + target available)
  5. Budget optimisation (if MMM fitted)
  6. What-if simulation (if MMM fitted)

All steps are capability-gated: if the dataset doesn't support a step,
it is skipped with a clear reason in ``unavailable_analyses``.  No step
ever falls back to demo data.

This module is called by ``analysis_runner.py`` — it accepts a
``CanonicalDataset`` and ``AnalysisContext`` and returns a ``Phase6Result``.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_engine.canonical import CanonicalDataset, AnalysisContext
from src.marketing.contracts import (
    Phase6Result, MarketingCapabilityReport, MarketingReadiness,
)
from src.marketing.field_discovery import assess_marketing_capabilities
from src.marketing.mmm_engine import fit_mmm
from src.marketing.contribution import compute_contributions
from src.marketing.roi import compute_roi_naive, compute_roi_model_based
from src.marketing.optimizer import optimize_budget
from src.marketing.simulator import simulate_standard_scenarios

logger = logging.getLogger(__name__)


def run_marketing_analysis(
    canonical: CanonicalDataset,
    context: AnalysisContext,
    dataset_id: str = "",
    run_optimization: bool = True,
    run_whatif: bool = True,
) -> Phase6Result:
    """Run the full Phase 6 marketing intelligence pipeline.

    Parameters
    ----------
    canonical : the dataset's CanonicalDataset representation.
    context : the dataset's AnalysisContext (capabilities, quality).
    dataset_id : identifier for this dataset.
    run_optimization : whether to run budget optimisation.
    run_whatif : whether to run what-if simulation.

    Returns
    -------
    ``Phase6Result`` contract with all available analyses.
    """
    result = Phase6Result(dataset_id=dataset_id)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Field discovery & capability assessment
    # ------------------------------------------------------------------
    try:
        report = assess_marketing_capabilities(canonical, context)
    except Exception as exc:
        logger.error("Field discovery failed: %s", exc)
        report = MarketingCapabilityReport()
        report.readiness = MarketingReadiness.NOT_READY
        report.unavailable_analyses.append(
            f"Field discovery error: {exc}")

    result.marketing_report = report

    df = canonical.dataframe
    if df is None or df.empty:
        result.unavailable_analyses.append(
            "No data available for marketing analysis.")
        return result

    channels = report.channel_fields
    target_col = report.target_field
    time_col = report.time_field
    control_cols = report.control_fields

    # ------------------------------------------------------------------
    # Step 2: ROI/ROAS (available if channels + target)
    # ------------------------------------------------------------------
    if report.roi.available and target_col and channels:
        try:
            roi_results = compute_roi_naive(df, channels, target_col)
            result.roi_results = roi_results
        except Exception as exc:
            logger.warning("Naive ROI failed: %s", exc)
            result.unavailable_analyses.append(
                f"ROI calculation error: {exc}")
    else:
        if report.roi.reason:
            result.unavailable_analyses.append(report.roi.reason)

    # ------------------------------------------------------------------
    # Step 3: MMM (requires channels + target + time + enough rows)
    # ------------------------------------------------------------------
    if report.mmm.available and target_col and channels:
        try:
            mmm_result = fit_mmm(
                df=df,
                channels=channels,
                target_col=target_col,
                control_cols=control_cols,
                time_col=time_col,
            )
            result.mmm_result = mmm_result

            # If MMM fitted successfully, compute model-based extras
            if mmm_result._model is not None:
                # Step 4: Channel contribution
                try:
                    contributions = compute_contributions(df, mmm_result)
                    result.contributions = contributions

                    # Upgrade ROI to model-based
                    try:
                        roi_model = compute_roi_model_based(
                            df, channels, target_col, contributions)
                        result.roi_results = roi_model
                    except Exception as exc:
                        logger.warning("Model-based ROI failed: %s", exc)

                except Exception as exc:
                    logger.warning("Channel contribution failed: %s", exc)
                    result.unavailable_analyses.append(
                        f"Channel contribution error: {exc}")

                # Step 5: Budget optimisation
                if run_optimization and report.budget_optimization.available:
                    try:
                        opt_result = optimize_budget(df, mmm_result)
                        result.optimization_result = opt_result
                    except Exception as exc:
                        logger.warning("Budget optimisation failed: %s", exc)
                        result.unavailable_analyses.append(
                            f"Budget optimisation error: {exc}")

                # Step 6: What-if simulation
                if run_whatif and report.what_if.available:
                    try:
                        whatif_result = simulate_standard_scenarios(
                            df, mmm_result)
                        result.whatif_result = whatif_result
                    except Exception as exc:
                        logger.warning("What-if simulation failed: %s", exc)
                        result.unavailable_analyses.append(
                            f"What-if simulation error: {exc}")
            else:
                result.unavailable_analyses.append(
                    "MMM fitting completed but no model was produced. "
                    + "; ".join(mmm_result.limitations))

        except Exception as exc:
            logger.error("MMM fitting failed: %s", exc)
            result.unavailable_analyses.append(
                f"MMM fitting error: {exc}")
    else:
        if report.mmm.reason:
            result.unavailable_analyses.append(report.mmm.reason)

    elapsed = time.time() - t0
    logger.info("Phase 6 completed in %.1fs (readiness=%s, channels=%d)",
                elapsed, report.readiness.value, len(channels))

    return result
