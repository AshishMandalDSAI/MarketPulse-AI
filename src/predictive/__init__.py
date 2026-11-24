"""Module 16 - Predictive Analytics (Phase 8 Part 4). Streamlit-free engine package.

Stages (see docs/PHASE_8_PART_4_REPORT.md):
  P0 pre-split preparation (duplicates, cap, split) - target-independent / count-only
  P1 CV-fold training-only screening and fitting (ScreenedModel.fit on T_f)
  P2 fold scoring, selection by cross-validation only
  P3 final full-training leakage review and refit (T only)
  P4 one-time held-out evaluation of the selected model
"""
