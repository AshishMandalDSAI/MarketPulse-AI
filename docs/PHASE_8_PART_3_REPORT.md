# Phase 8 Part 3 - Module 15: Inferential Statistics

Status: implemented and delivered for independent review. **Not accepted** - acceptance belongs to the independent reviewer.
Baseline: `MarketPulse_AI_Phase8_Part2_Descriptive_Statistics_FINAL.zip`, SHA-256
`38624ed518efdf82e78e9abe4a9ff64d84524f8987507204698db6a59c9bd528` (verified before any work started).

## 1. What was built

| Area | File |
|---|---|
| Result contracts | `src/inferential/contracts.py` |
| Assumption layer | `src/inferential/assumptions.py` |
| Nine tests plus the mean confidence interval | `src/inferential/tests.py` |
| Effect sizes | `src/inferential/effect_size.py` |
| Interpretation layer | `src/inferential/interpret.py` |
| CSV / Excel export | `src/inferential/export.py` |
| Dashboard body | `dashboard/components/inferential_ui.py` |
| Page | `dashboard/pages/15_inferential_statistics.py` (passes the body to the shared scaffold) |

The page reuses `render_module_page` from the shared scaffold; no descriptive functionality is duplicated.

### Tests and the "tenth item"
Nine named tests: Z, one-sample t, independent t, paired t, one-way ANOVA, chi-square goodness-of-fit, chi-square
independence, Pearson significance, Spearman significance. The original specification lists ten Module 15 items; items
"Hypothesis Testing" and "Confidence Intervals" are capabilities, not separate tests. They are delivered as (a) a shared
H0 / H1 / decision framework and a confidence interval on every test, and (b) a standalone "Confidence interval for a mean"
tool. **No test was invented.** The reviewer was asked about this once and expressed no preference, so this documented
default was applied.

## 2. Methods (SciPy does every calculation)

| Test | SciPy call | Statistic / df | Effect size |
|---|---|---|---|
| One-sample t | `ttest_1samp` | t, n-1 | Cohen's d (Hedges' g in details) |
| Z (σ supplied) | `norm.sf/cdf` | z, none | Cohen's d = (x̄-μ0)/σ |
| Independent t | `ttest_ind` (Welch default; Student optional) | t, Welch-Satterthwaite or n1+n2-2 | Cohen's d (pooled SD) |
| Paired t | `ttest_rel` | t, pairs-1 | Cohen's d_z |
| One-way ANOVA | `f_oneway` | F, k-1 and N-k | η² (ω² in details) |
| Chi-square GOF | `chisquare` | χ², categories-1 | Cohen's w |
| Chi-square independence | `chi2_contingency` (Yates for 2x2) | χ², (r-1)(c-1) | Cramér's V (from the uncorrected χ²) |
| Pearson | `pearsonr` | r, n-2 | r |
| Spearman | `spearmanr` | rho, n-2 **(approximate: df of SciPy's Student-t approximation, not an exact property of rho; see the final correction)** | rho |

- Significance is strict: `p < α`. `p == α` is **not** significant. α is selectable (0.01 / 0.05 / 0.10; default 0.05).
- Intervals are always two-sided, whatever the alternative hypothesis, and are produced by a separate SciPy call.
  Level is user-selectable (90 / 95 / 99%) with the correct df.
- Pearson CI: Fisher z (SciPy). Spearman CI: Fisher z with the Bonett-Wright standard error; disclosed on the page as an
  approximation and undefined (with a reason) below 10 pairs or for |rho| = 1.
- ANOVA and chi-square have no single interval; the interval is blank with the reason shown, never 0.
  ANOVA reports a per-group mean interval in a table.

## 3. Assumption layer
Checked before any statistic is calculated: sample size, normality (Shapiro-Wilk up to n = 5,000, D'Agostino above),
equal variances (Levene, median-centred), independence (time-ordered data is flagged; lag-1 autocorrelation is reported; for the two chi-square tests a lag-1 transition-table screen is used - see the final correction),
constant values, missing data, invalid numeric input (text, dates, booleans, infinities), chi-square expected counts,
paired requirements and data types. Each violation states why it matters, labels the result **invalid** (refused),
**unreliable** or **usable with caution**, and recommends an alternative. An unsuitable test is never run silently.

- **Chi-square:** refused when Cochran's rule fails (any expected count below 1, or more than 20% of cells below 5).
  Exactly 20% is allowed with a caution. The expected-count table is still shown so the user sees why.
- **Z-test:** allowed only when the user supplies a positive finite population SD **and** (n >= 30 **or** normality passes).
  Otherwise refused with a pointer to the one-sample t-test. The sample SD is never substituted.
- **Numeric columns that are really categories:** a numeric column with more than 12 distinct values is not treated as a
  category for chi-square; the test is refused with that reason.

## 4. Missing data
Listwise deletion per test (the dataset itself is never changed). Original, removed and final counts and the missing % are
shown and exported; per-column counts are given; blank text counts as missing. Nothing is imputed, nothing is zero-filled.
For group tests, complete rows that belong to groups not being compared are reported separately as "not used".

## 5. Interpretation layer
Every result includes: what the p-value means (and is not), the decision (reject / fail to reject, never "accept H0"),
statistical significance, what the result does **not** prove, and practical importance (statistical vs practical
significance, with explicit wording for tiny effects with large samples and large effects with small samples).
Every correlation output states "Correlation does not establish causation."

## 6. Dashboard
Sections: Module Overview, Dataset Information, Statistical Test Selection, Input Controls, Assumption Checks,
Statistical Results, Effect-Size Interpretation, Confidence Intervals, Plain-English Explanation, Downloads.
No column name is hard-coded; every control is built from the active dataset's own column metadata. Tests the dataset
cannot support are listed with the reason. Nothing is shown as a result unless it was calculated from the active dataset.
Identifier and empty columns are not offered as inputs.

## 7. Isolation and caching
- All session state uses the `mp_page_` convention (`src/common/session_scope.py`); widget keys include the dataset id.
- **There is deliberately no result cache.** Tests run live on every rerun (a t-test on 200,000 rows takes 0.06 s;
  the slowest, chi-square on 200,000 rows, 0.36 s), so a stale result cannot occur by construction.
  Module-level state holding results does not exist (asserted by a test).
- Streamlit drops the state of widgets that are not rendered. After switching datasets and back, the selected test returns
  to its default; this is normal Streamlit behaviour, not a result leak.
- The Part 1 cache (`data_loader.py`) and the Part 2 module are untouched.

## 8. Exports
CSV (single table), CSV zip (all tables) and Excel workbook. Sheets: Test Summary (35 columns), Assumption Checks, Details,
Methodology. File names are `inferential_statistics_<dataset>[_<table>]`. Undefined values are blank with the reason in a
neighbouring column. All text goes through the shared `sanitize_for_spreadsheet` (formula-injection protection).
The CSV is byte-for-byte reproducible; the zip and xlsx embed file timestamps.

## 9. Shared-file changes (all deliberate)
| File | Change | Reason |
|---|---|---|
| `src/common/modules.py` | `implemented=True` on the inferential spec (one added line) | the registry must report real status |
| `dashboard/pages/15_inferential_statistics.py` | passes the body function to the scaffold | was a placeholder |
| `tests/test_phase8_part2_dashboard.py` | `test_only_the_descriptive_module_is_implemented` now expects exactly `["descriptive", "inferential"]` | the old assertion ("only descriptive") is false by design after Part 3; it stays exact, so modules 16-17 must still be unimplemented. Not weakened. |
| `docs/PART_STATUS.md` | **new** (did not exist); one Part 3 line | records delivery only, never acceptance |

## 10. Limitations
- Independence cannot be verified from data; the page says so on every test. For the chi-square tests, time-ordered rows carry an explicit
  "independence NOT established" warning and are downgraded to *unreliable* when a lag-1 screen detects serial dependence (final correction).
- Spearman's degrees of freedom (n - 2) are approximate (Student-t approximation behind SciPy's p-value) and are labelled so everywhere (final correction).
- No multiple-testing correction; stated on the page.
- No non-parametric alternatives (Mann-Whitney, Wilcoxon, Kruskal-Wallis, Fisher's exact) are implemented - they are
  named as recommendations only, because none was in scope and none was invented.
- The Spearman interval is approximate.
- Effect-size labels follow conventional rules of thumb (Cohen); the page says they are not business thresholds.
- AppTest simulates sessions in one process; it is not a live multi-browser test.

## 11. Verification
See `PHASE_8_PART_3_TEST_REPORT.md` for exact commands and outputs.
