# PART 02 optimizer default-limit fix

## The bug

`dashboard/components/prescriptive_ui.py::_constraint_inputs` (used by **15 · Prescriptive Analytics**) gave every channel these defaults:

- a minimum of 5% of the total;
- a maximum of **55% of the total**.

With one channel, the maximums add up to 55% of the budget. The optimiser then refused before the user had changed anything: *"The channel maximums add up to only 124,291.37, less than the total budget 225,984.30."*

The 5% minimum had the mirror problem: with more than 20 channels, the minimums add up to more than the budget.

The widgets were also created with `value=` defaults. Changing the total therefore left stale defaults in the session.

## The fix

### New module: `src/prescriptive/limits.py` (pure Python, unit-tested)

**`default_limits(total, channels, current)`** gives each of the *n* channels:

| | Default |
|---|---|
| Minimum | `min(5%, 50%/n)` of the total, never more than half the channel's current share; **0** for a single channel |
| Maximum | **100%** of the total for 1–2 channels; otherwise `max(55%, 200%/n)`, raised to 125% of the channel's current share when that's higher, capped at 100% |

So `sum(min) ≤ total ≤ sum(max)` holds **by construction**. Rounding is then corrected so the guarantee still holds to the cent.

**`reconcile_limits(state, pairs, mins, maxs, registry)`** runs before the widgets are created:

- It records the last value it wrote automatically into each widget key.
- If a widget still holds that value, it's refreshed when the total changes.
- If the user typed a different value, it's **never overwritten**, and it's labelled "· set by you".

**`reset_limits`** is the only way to replace user values. It's an explicit button, "Reset all limits to the defaults", which appears only when a value has been edited.

**`feasibility(total, mins, maxs, channels)`** shows `sum of minimums | total | sum of maximums` and a plain explanation:

- **Infeasible:** e.g. "The maximums add up to only X, less than the total Y. Raise some maximums by Z in total, or lower the total."
- **Feasible:** "Feasible: minimums ≤ total ≤ maximums".

When the limits are infeasible, the **Find the best allocation** button is disabled. Nothing is changed silently, and no allocation is invented.

**Solver status and objective** are always shown with a result (`_solver_report`):

- status: Solved / Infeasible constraints / Solver failed / Not run;
- whether the solver converged, and its message;
- the objective in words;
- the method.

### Where the fix applies

- **15 · Prescriptive Analytics** (the original location of the bug).
- **06 · Marketing Resource Optimizer**, spend mode: the same `_constraint_inputs`, with per-page keys and the currency unit.
- **06 · Marketing Resource Optimizer**, exposure mode: the same `default_limits`, `reconcile_limits` and `feasibility`, in exposure units ("not money").

The engine's own check (`engine.validate_constraints`) is unchanged and still runs before solving. The previous Budget Optimizer page (kept, unchanged, behind its switch) uses its own engine (`src/optimization`), with bounds derived from historical spend, not the 55% screen. PART 04 reviews it with the full constraint engine.

## Evidence (tests actually run; results in `PART02_TEST_LOG.txt`)

### Unit tests: `TestDefaultLimits`, 31 tests

- **Feasibility for every combination:** defaults are feasible for 1, 2, 3, 4, 5, 10, 25 and 60 channels × totals of 1, 225,984.30 and 1e9.
- **The reported case:** one channel with a total of 225,984.30. The old 55% maximum reproduces the error; the new defaults give max = total, min = 0, and validation passes.
- **No fixed 55%:** the maximum is 100% for 1 and 2 channels, and a channel holding 70% of current spend isn't forced below 87.5%.
- **Refresh and manual values:** `reconcile_limits` refreshes automatic values on a budget change, keeps a typed value across two budget changes, and only `reset_limits` replaces it.
- **Messages:** the infeasible and feasible messages are checked.
- **No 55% left in the screen code:** `prescriptive_ui.py` no longer contains `0.55`.

### App tests: `TestOptimizerPages`, real Streamlit pages on uploaded datasets

- **One, two and four channels:** the default limits are feasible, the button is enabled, and the result says "Solver status: Solved" with the objective.
- **Two channels, budget changed:** after a maximum is typed and the total doubles, the untouched maximum follows the total, the typed maximum stays at 1,234 (marked "set by you"), and the reset button appears.
- **Four channels, all maximums set to 1.0:** the error "maximums add up to only 4.00" is shown, the values aren't changed, the button is disabled and no result appears. Pressing reset makes the limits feasible.
- **Page 15, one channel:** maximum ≥ total, and the result is Solved.
- **Exposure optimizer:** Solved, in impressions.

### Browser (Chrome)

One-channel upload, page 06: "Solver status: Solved · converged: yes".

## Left for PART 04 (as agreed)

- reserve or unallocated-budget options;
- full constraint diagnostics and repair suggestions (actions A–D);
- sensitivity analysis;
- extended solver reporting;
- the advanced optimizer test battery.
