"""Upload Data page — the entry point for the dataset-first platform.

Workflow (each step visible to the user, nothing changed silently):

1. Read the file (CSV / XLSX) with friendly error messages.
2. Assess data quality on the *original* data: missing, blank and placeholder
   cells, invalid numbers/dates, infinities, duplicates, mixed types, and an
   explainable quality score.
3. Detect column roles. HIGH-confidence roles are auto-selected; MEDIUM/LOW
   roles need explicit confirmation.
4. Review a recommended cleaning plan (per column, with reasons), edit it,
   and preview the before/after effect. The original file is never modified.
5. Confirm & Analyze: the original bytes, the cleaned data and a full audit
   trail are stored per dataset, then the analysis pipeline runs.
"""

import copy
import hashlib
import io
import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import DEMO_DATASET_ID, dataset_raw_path, dataset_processed_dir, ensure_dataset_dirs
from dashboard import navigation
from dashboard.components import session_paths
from dashboard.components import role_mapping_ui as RMU
from dashboard.components.kpi_card import render_kpi_row
from dashboard.components.ui import callout, empty_state, page_header, section, show_table
from src.data_engine.pipeline import profile_and_map, confirm_and_standardize
from src.data_engine.capability_engine import detect_capabilities
from src.data_engine.analysis_runner import run_full_analysis, normalize_duplicate_columns
from src.common.formatting import escape_text
from src.data_engine import role_engine as RE
from src.data_engine.currency import CURRENCIES, choice_to_info, detect_currency, save_currency
from src.data_engine.data_quality import (STRATEGIES, SCORE_WEIGHTS, apply_cleaning, assess_quality,
                                         recommend_plan)
from src.kpi.generic_kpis import compute_generic_kpis

BADGE = {"HIGH": "🟢 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🔴 LOW"}
_NUMERIC_ONLY = {"mean", "median", "zero", "ffill", "past_median", "interpolate"}
_CATEGORICAL_ONLY = {"mode", "unknown"}


class UploadReadError(Exception):
    """A file could not be read; the message is safe to show to the user."""


def _load_dataframe(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not data or not data.strip():
        raise UploadReadError("The file is empty.")
    if suffix == ".csv":
        last_error = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=encoding, sep=None, engine="python")
            except UnicodeDecodeError as e:
                last_error = e
            except pd.errors.EmptyDataError:
                raise UploadReadError("The CSV file has no columns or header row.")
            except pd.errors.ParserError:
                raise UploadReadError("The CSV structure could not be read — rows have inconsistent numbers of "
                                      "fields. Check for unquoted commas inside values.")
        raise UploadReadError(f"Unsupported text encoding ({type(last_error).__name__}). Save the file as UTF-8 CSV.")
    if suffix == ".xlsx":
        try:
            return pd.read_excel(io.BytesIO(data))
        except Exception:
            raise UploadReadError("This doesn't look like a valid Excel (.xlsx) workbook. Open it in Excel and "
                                  "save it again as .xlsx, or export it as CSV.")
    raise UploadReadError("Only .csv and .xlsx files are supported. Save legacy .xls files as .xlsx first.")


def _types_only_plan(report: dict) -> dict:
    """Lossless normalisation used for column detection: blanks -> missing,
    numbers stored as text -> numeric. No values are imputed."""
    return {"columns": {c: {"strategy": "none", "convert_numeric": v["kind"] == "numeric_text"}
                        for c, v in report["columns"].items()},
            "drop_duplicate_rows": False}



def _go_to_overview() -> None:
    st.session_state["mp_force_page"] = navigation.EXECUTIVE_OVERVIEW

@st.cache_data(show_spinner=False, max_entries=4)
def _analyse_file(file_hash: str, name: str, _data: bytes):
    """Read + assess + type-normalise + map. Cached per file content hash."""
    class _F:  # minimal UploadedFile stand-in for _load_dataframe
        def __init__(self, n, d):
            self.name, self._d = n, d

        def getvalue(self):
            return self._d
    df = _load_dataframe(_F(name, _data))
    df, dup_map = normalize_duplicate_columns(df)
    report = assess_quality(df)
    normalised, _ = apply_cleaning(df, _types_only_plan(report))
    profile, mapping = profile_and_map(normalised)
    return df, dup_map, report, normalised, profile, mapping


def _score_pill(score: float) -> str:
    if score >= 90:
        return "Good"
    if score >= 75:
        return "Fair"
    return "Needs attention"


def _render_quality(report: dict, dup_map: dict) -> None:
    section("1 · What we found")
    render_kpi_row([
        {"label": "Rows", "value": f"{report['n_rows']:,}"},
        {"label": "Columns", "value": f"{report['n_cols']:,}"},
        {"label": "Data quality score", "value": f"{report['score']:.1f} / 100",
         "help": _score_pill(report["score"])},
        {"label": "Missing cells", "value": f"{report['missing_pct']:.1f}%",
         "help": f"{report['missing_cells']:,} of {report['total_cells']:,} cells"},
        {"label": "Exact duplicate rows", "value": f"{report['duplicate_rows']:,}",
         "help": "Flagged, kept unless you choose to remove them"},
    ])
    issues = []
    if report["blank_string_cells"]:
        issues.append(f"{report['blank_string_cells']:,} cell(s) are empty text, whitespace or placeholders "
                      "such as 'N/A' — they are treated as missing.")
    if report["invalid_cells"]:
        issues.append(f"{report['invalid_cells']:,} value(s) are invalid (unreadable numbers, invalid dates or "
                      "infinities).")
    if dup_map:
        issues.append("Duplicate column names were renamed: "
                      + ", ".join(f"{v!r} → {k!r}" for k, v in dup_map.items()) + ".")
    if report["rows_all_missing"]:
        issues.append(f"{report['rows_all_missing']:,} row(s) are completely empty.")
    for msg in issues:
        callout(msg, kind="warn")

    with st.expander("How the quality score is calculated"):
        dims = report["dimensions"]
        st.markdown(report["score_method"])
        show_table(pd.DataFrame([
            {"Dimension": k.replace("_", " ").title(), "Score": dims[k], "Weight %": w * 100}
            for k, w in SCORE_WEIGHTS.items()
        ]), decimals=1)
        st.caption("Imputed values never count as complete: after cleaning, the score is reported "
                   "separately and filled cells are listed as imputed.")

    rows = []
    for col, c in report["columns"].items():
        if c["n_missing"] or c["n_invalid"] or c["mixed_types"] or not c["storage_matches_content"]:
            rows.append({"Column": col, "Detected type": c["kind"].replace("_", " "),
                         "Missing": c["n_missing"], "Missing %": c["missing_pct"],
                         "Invalid": c["n_invalid"], "Notes": " ".join(c["notes"])})
    if rows:
        st.markdown("**Missing and invalid values by column**")
        show_table(pd.DataFrame(rows), pct_cols=["Missing %"], int_cols=["Missing", "Invalid"])
        hist = report["missing_per_row_histogram"]
        st.caption(f"Rows by number of missing cells — none: {hist['0']:,} · one: {hist['1']:,} · "
                   f"two: {hist['2']:,} · three or more: {hist['3+']:,}.")
    else:
        callout("No missing or invalid values were found.", kind="good")


def _role_map(mapping, confirmed_roles, overrides) -> tuple:
    date_col = overrides.get("date_column", mapping.date.column if "date" in confirmed_roles else None)
    revenue_col = overrides.get("revenue_column", mapping.revenue.column if "revenue" in confirmed_roles else None)
    roles = {}
    if revenue_col:
        roles[revenue_col] = "revenue"
    for ch in overrides.get("channel_columns", []):
        roles[ch] = "channel"
    if date_col:
        roles[date_col] = "date"
    return date_col, roles


def _monetary_columns(mapping, roles: dict, detected) -> list:
    """Columns whose values are money: confirmed revenue and spend channels, mapped
    cost / profit, and any column that itself carries currency evidence."""
    cols = [c for c, r in roles.items() if r in ("revenue", "channel")]
    for role in ("cost", "profit"):
        g = getattr(mapping, role, None)
        if g is not None and getattr(g, "column", None):
            cols.append(g.column)
    for line in (detected.evidence if detected else []):
        cols.append(line.split(":", 1)[0])
    return list(dict.fromkeys(c for c in cols if c))


@st.cache_data(show_spinner=False, max_entries=8)
def _detect_currency_cached(file_hash: str, _df: pd.DataFrame):
    return detect_currency(_df)


def _render_currency(df: pd.DataFrame, file_hash: str):
    """Step 2b: automatic currency detection with a manual override (PART 01)."""
    section("2b · Currency")
    detected = _detect_currency_cached(file_hash, df)
    if detected.identified:
        what = f"**{detected.code}** ({detected.symbol})" if detected.code else f"the symbol **{detected.symbol}**"
        st.markdown(f"Detected {what} from {detected.source}. " + (detected.note or ""))
    else:
        st.markdown("No currency could be identified reliably, so **no currency symbol will be shown** "
                    "unless you choose one. " + (detected.note or ""))
    if detected.evidence or detected.conflicts:
        with st.expander("Evidence"):
            for line in detected.evidence:
                st.markdown(f"- {escape_text(line)}")
            for line in detected.conflicts:
                st.markdown(f"- Conflict: {escape_text(line)}")
    auto_label = (f"Auto-detected: {detected.code or detected.symbol}" if detected.identified
                  else "Auto-detected: none (no symbol)")
    options = ["auto", "none"] + sorted(CURRENCIES)
    choice = st.selectbox(
        "Currency of the monetary columns", options, key=f"currency_choice_{file_hash[:12]}",
        format_func=lambda c: auto_label if c == "auto" else (
            "None / not monetary (no symbol)" if c == "none" else f"{c} · {CURRENCIES[c][1]}"),
        help="Used for KPI cards, tables and chart axes. MarketPulse AI never assumes a currency.")
    return choice, detected


def _render_cleaning(df: pd.DataFrame, report: dict, date_col, roles, file_hash: str):
    section("3 · Data cleaning")
    plan = recommend_plan(df, report, date_column=date_col, roles=roles)
    rows = []
    for col, p in plan["columns"].items():
        c = report["columns"].get(col, {})
        if p["strategy"] == "none" and not p.get("convert_numeric"):
            continue
        rows.append({"Column": col, "Type": c.get("kind", "").replace("_", " "),
                     "Missing": c.get("n_missing", 0), "Invalid": c.get("n_invalid", 0),
                     "Treatment": STRATEGIES[p["strategy"]], "Why": p["reason"]})

    if not rows and not report["duplicate_rows"]:
        callout("Nothing needs cleaning — the data will be analyzed as uploaded.", kind="good")
        return plan

    if rows:
        st.caption("Recommended treatment per column. Change any row before analyzing; the original "
                   "file is stored unchanged either way.")
        labels = list(STRATEGIES.values())
        edited = st.data_editor(
            pd.DataFrame(rows), hide_index=True, use_container_width=True,
            key=f"cleaning_plan_{file_hash[:12]}",
            disabled=["Column", "Type", "Missing", "Invalid", "Why"],
            column_config={
                "Treatment": st.column_config.SelectboxColumn("Treatment", options=labels, required=True,
                                                              width="medium"),
                "Why": st.column_config.TextColumn("Why", width="large"),
            },
        )
        by_label = {v: k for k, v in STRATEGIES.items()}
        for _, r in edited.iterrows():
            col = r["Column"]
            chosen = by_label.get(r["Treatment"], "leave")
            kind = report["columns"].get(col, {}).get("kind", "")
            numeric = kind in ("numeric", "numeric_text")
            if (chosen in _NUMERIC_ONLY and not numeric) or (chosen in _CATEGORICAL_ONLY and numeric):
                st.warning(f"'{STRATEGIES[chosen]}' does not fit column '{col}' ({kind}); it will be left as is.")
                chosen = "leave"
            if chosen in ("ffill", "past_median", "interpolate") and not date_col:
                st.warning(f"'{STRATEGIES[chosen]}' needs a confirmed date column; '{col}' will be left as is.")
                chosen = "leave"
            plan["columns"][col]["strategy"] = chosen

    if report["duplicate_rows"]:
        plan["drop_duplicate_rows"] = st.checkbox(
            f"Remove {report['duplicate_rows']:,} exact duplicate row(s)", value=False,
            help=plan["duplicate_reason"], key=f"drop_dups_{file_hash[:12]}")

    cleaned, audit = apply_cleaning(df, plan)
    after = assess_quality(cleaned, imputed_cells=audit["total_imputed_cells"])
    st.markdown("**Preview of the effect**")
    render_kpi_row([
        {"label": "Quality score", "value": f"{report['score']:.1f} → {after['score']:.1f}",
         "help": "Original → after cleaning (imputed cells are not counted as complete)"},
        {"label": "Cells imputed", "value": f"{audit['total_imputed_cells']:,}"},
        {"label": "Rows removed", "value": f"{audit['total_rows_removed']:,}",
         "help": f"{len(df):,} → {len(cleaned):,} rows"},
        {"label": "Converted / normalised", "value": f"{len(audit['columns_converted_to_numeric'])} col · "
                                                     f"{audit['blank_strings_normalised']:,} blanks",
         "help": "Numbers stored as text converted; blank/placeholder cells marked missing"},
    ])
    changed_cols = [c for c in audit["transformed_columns"] if c in df.columns and c in cleaned.columns]
    if changed_cols:
        common = cleaned.index.intersection(df.index)
        before_vals = df.loc[common, changed_cols].astype(str)
        after_vals = cleaned.loc[common, changed_cols].astype(str)
        diff_rows = (before_vals != after_vals).any(axis=1)
        sample_idx = list(common[diff_rows.to_numpy()][:15])
        if sample_idx:
            with st.expander(f"Before / after for changed rows (showing {len(sample_idx)})"):
                b = df.loc[sample_idx, changed_cols].add_suffix(" (before)")
                a = cleaned.loc[sample_idx, changed_cols].add_suffix(" (after)")
                cols_order = [x for pair in zip(b.columns, a.columns) for x in pair]
                st.dataframe(pd.concat([b, a], axis=1)[cols_order].astype(str), use_container_width=True)
    st.download_button("⬇️ Download cleaned data (CSV)", cleaned.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_data.csv", mime="text/csv", key=f"dl_clean_{file_hash[:12]}")
    st.session_state["mp_pending_cleaning"] = (plan, audit, after)
    return plan


def render():
    page_header("Step 1 · Data", "Upload data",
                "Bring any CSV or Excel (.xlsx) file with a header row. MarketPulse AI detects what each "
                "column contains, shows the data's quality, and lets you approve every correction before "
                "analysis. Your original file is stored unchanged.")

    if st.button("▶️ Or just try the built-in demo dataset"):
        session_paths.reset_to_demo_dataset()
        st.session_state["mp_force_page"] = navigation.EXECUTIVE_OVERVIEW
        st.rerun()

    uploaded_file = st.file_uploader(
        "CSV or Excel (.xlsx) file", type=["csv", "xlsx"],
        help="Legacy .xls workbooks are not supported — save them as .xlsx first. Up to 200 MB.",
    )
    if uploaded_file is None:
        empty_state("No file uploaded yet",
                    "Supported: **.csv** (UTF-8 or Latin-1, comma/semicolon/tab separated) and **.xlsx** "
                    "(first sheet). One header row, one record per row; column names can be anything.",
                    "For marketing analyses include a date, a revenue column and spend columns per channel.")
        return

    data = uploaded_file.getvalue()
    file_hash = hashlib.sha256(data).hexdigest()
    with st.status("Reading and checking your file…", expanded=False) as status:
        try:
            df, dup_map, report, normalised, profile, mapping = _analyse_file(file_hash, uploaded_file.name, data)
        except UploadReadError as e:
            status.update(label="The file could not be read", state="error")
            st.error(str(e))
            return
        except Exception as e:  # unexpected: keep the message short, never a traceback
            status.update(label="The file could not be read", state="error")
            st.error(f"This file could not be processed ({type(e).__name__}). Check that it is a valid CSV/XLSX.")
            return
        status.update(label=f"Checked {len(df):,} rows × {df.shape[1]} columns", state="complete")

    if df.empty:
        st.warning("This file has a header but no data rows.")
        return

    _render_quality(report, dup_map)
    mapping = copy.deepcopy(mapping)  # cached object must never be mutated
    # PART 02: the universal business-role mapping replaces the earlier role widgets. It covers the same
    # roles (date, revenue/target, channels) plus exposure, price, promotion, trade spend, dimensions...
    # and feeds the existing pipeline through the same overrides it always accepted.
    section("2 · Column roles and target")
    role_map, roles_ready = RMU.render_upload_roles(normalised, file_hash, uploaded_file.name)
    confirmed_roles, overrides = RE.legacy_overrides(role_map)
    date_col, roles = _role_map(mapping, confirmed_roles, overrides)
    currency_choice, currency_detected = _render_currency(df, file_hash)
    plan = _render_cleaning(df, report, date_col, roles, file_hash)

    section("4 · Analyze")
    if not roles_ready:
        st.warning("Confirm the analysis target(s) in step 2 before analysing.")
    if st.button("✅ Confirm & Analyze", type="primary", disabled=not roles_ready):
        # Phase 1: ALWAYS generate a fresh UUID. Never reuse mp_pending_dataset_id
        # or any previous ID. Even re-uploading the same file gets a new ID.
        dataset_id = f"upload_{uuid.uuid4().hex[:12]}"

        with st.status("Analyzing…", expanded=True) as status:
            st.write("Saving the original file unchanged")
            ensure_dataset_dirs(dataset_id)
            safe_name = Path(uploaded_file.name).name.replace("/", "_").replace("\\", "_") or "upload.csv"
            dataset_raw_path(dataset_id, safe_name).write_bytes(data)

            st.write("Applying the approved cleaning plan")
            cleaned, audit = apply_cleaning(df, plan)
            after = assess_quality(cleaned, imputed_cells=audit["total_imputed_cells"])
            audit["source_file"] = safe_name
            audit["source_sha256"] = file_hash
            quality = {"original": report, "cleaned": after, "audit": audit}

            st.write("Running models and analytics")
            result = run_full_analysis(dataset_id, df=cleaned, mapping=copy.deepcopy(mapping),
                                       confirmed_roles=confirmed_roles, overrides=overrides,
                                       data_quality=quality)
            money_cols = list(dict.fromkeys(_monetary_columns(mapping, roles, currency_detected)
                                            + RE.monetary_columns(role_map)))
            save_currency(dataset_processed_dir(dataset_id),
                          choice_to_info(currency_choice, currency_detected, money_cols))
            RMU.save_upload_mapping(dataset_id, role_map, uploaded_file.name)
            session_paths.set_active_dataset(dataset_id, filename=uploaded_file.name)
            status.update(label=f"Analysis complete — dataset {dataset_id}", state="complete")

        st.success(f"Dataset analyzed as `{dataset_id}`. Use the sidebar to explore the results.")
        section("Available analyses")
        for cap in result["capabilities"].capabilities:
            icon = "✅" if cap.available else "⛔"
            st.markdown(f"{icon} **{cap.name}** — {cap.reason}")

        kpis = result["generic_kpis"]
        if kpis:
            section("KPI preview")
            cols = st.columns(min(4, len(kpis)))
            for i, kpi in enumerate(kpis[:4]):
                cols[i % len(cols)].metric(kpi.label, kpi.value)
            with st.expander("All generic KPIs"):
                for kpi in kpis:
                    st.write(f"**{kpi.label}**: {kpi.value} — {kpi.note}")

        with st.expander("Processing log and mapping decisions"):
            for line in result["report"]:
                st.write(f"- {line}")
        st.download_button("⬇️ Download cleaning audit (JSON)",
                           json.dumps(audit, indent=2, default=str).encode("utf-8"),
                           file_name=f"{dataset_id}_cleaning_audit.json", mime="application/json")

        # Nested inside the Confirm button's branch, a plain `if st.button(...)` never
        # fires (the rerun it triggers has Confirm == False). A callback runs first.
        st.button("Go to Executive Overview →", key="mp_goto_overview_after_upload",
                  on_click=_go_to_overview)
