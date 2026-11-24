# Architecture: upgrade PARTs 01–05

The baseline architecture (`docs/architecture.md`) is unchanged: a Streamlit dashboard
(`app.py`, `dashboard/`), a FastAPI service (`api/main.py`, unchanged by the upgrade), and the
analytics packages under `src/`. The upgrade adds modules beside the existing ones. It does not
replace them.

```text
Upload ─► src/data_engine/pipeline.py (profiling, baseline)
        └► src/data_engine/role_engine.py + role_lexicon.py   [PART 02]
              roles · confidence · evidence · targets · analytics mode (spend / exposure / general)
              role_mapping.json (versioned) in the dataset's processed dir
        └► src/data_engine/currency.py (detect / choose / none)          [PART 01]
        └► src/data_engine/analysis_runner.py (baseline Phase 5/6 outputs)

Pages 04–07 ─► dashboard/components/model_page.py (shared frame, session memo)  [PART 02]
               src/marketing_intel/
                 context.py       ModelContext: target, channels, controls, dims, frequency
                 response.py      driver response model (OLS, HC3, log1p, fixed effects, holdout)
                 effectiveness.py MODEL 04
                 channels.py      MODEL 05 (ROI/ROAS only with channel-wise cost)
                 optimizer.py     MODEL 06 exposure optimisation + statuses
                 scenarios.py     MODEL 07 scenarios, budget %, sensitivity   [PART 04]
                 capabilities.py  supported / unsupported per model
                 stats.py         small stats helpers (BH correction, etc.)

Optimisation ─► src/prescriptive/engine.py   optimize_allocation(..., allow_reserve, monetary)
                src/prescriptive/limits.py   default / reconciled / reset limits   [PART 02]
                src/prescriptive/feasibility.py  diagnose → remedies A–D, classify status [PART 04]
                dashboard/components/prescriptive_ui.py  constraint panel, remedies, status, utilisation

Statistics ─► dashboard/pages/statistics.py (3 tabs)                          [PART 01]
              dashboard/components/inferential_ui.py  Z-test σ rule, formulas, labelled charts [PART 03]
              src/predictive/classifiers.py  12-classifier registry, availability    [PART 03]
              src/predictive/evaluate.py     CV selection; per-model holdout (_holdout_all)

Roles  ─► PART 06: role `quality_score` (creative / content / quality scores; business driver, never spend)

Charts ─► dashboard/styles/theme.py apply_chart_style(title, x_title, y_title, currency)
          dashboard/components/chart_labels.py  value/time axis titles with units   [PART 01]
```

## Design rules

- **No silent assumptions.** Ambiguous roles, targets and currencies need the user's choice.
  An unknown currency shows no symbol.
- **No fabricated money.** Exposure units are never converted to money. Trade or promotion
  spend is a business variable, not channel spend.
- **Selection versus evaluation.** Models are selected on cross-validation only. The holdout
  scores each model once and never drives the choice.
- **Additive changes to protected files.** Nine baseline files changed, each with a
  documented reason (`FINAL_CHANGELOG.md` § Protected files). The API is unchanged.
- **Per-process state.** The API stays at a single worker, as in the baseline.
