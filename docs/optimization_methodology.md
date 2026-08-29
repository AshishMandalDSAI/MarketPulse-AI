# MarketPulse AI — Budget Optimization Methodology

## Objective

Maximize predicted weekly revenue `f(x)` (from the fitted MMM) subject to:

```
maximize   f(x)
subject to sum(x_i) = total_budget
           lower_i <= x_i <= upper_i   for each channel i
           x_i >= 0
```

where `x_i` is channel *i*'s weekly spend and `lower_i`/`upper_i` are configurable fractional bounds of the
total budget (defaults: 5% floor, 55% ceiling per channel — prevents degenerate all-in-one-channel
solutions).

## Solver

`scipy.optimize.minimize` with `method="SLSQP"` (Sequential Least Squares Programming) — chosen because it
natively supports bound constraints + a nonlinear equality constraint, and the objective (adstock →
saturation → linear regression) is smooth and differentiable, which SLSQP handles well without needing
manually-derived gradients.

## Handling adstock's time-dependence

Adstock is inherently path-dependent (this week's effect depends on the whole spend history). Rather than
re-optimizing an entire multi-week spend path (a much harder, higher-dimensional problem outside this
project's scope), the optimizer:

1. Computes each channel's recent **adstock carry-in** — the portion of adstocked effect from weeks prior
   to the current one — from the last 8 weeks of actual spend.
2. Holds that carry-in fixed while optimizing **this week's own spend allocation** across channels.
3. Business controls (price, promotion, holiday, competitor/economic index) are held at their recent
   8-week average.

This is a standard simplification for weekly budget-reallocation recommendation tools and is explicitly
documented here rather than silently assumed.

## Output

- `data/processed/budget_optimization_result.csv` — per-channel current vs. recommended budget, change, %.
- `data/processed/budget_optimization_summary.json` — current/optimized predicted revenue, expected
  improvement %, current/optimized efficiency.

## Validation

`tests/test_mmm_pipeline.py::test_optimizer_allocation_sums_to_budget` asserts the optimized allocation sums
to the requested total budget (within numerical tolerance) and that no channel receives negative spend.
