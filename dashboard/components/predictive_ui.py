"""Module 16 (Predictive Analytics) sections. Results are computed only when the user presses Run, cached under the
active dataset's scope and re-run when any setting changes; nothing is shown from another dataset."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import session_paths
from dashboard.components.stats_ui import (apply_dark_layout, cache_get, cache_set, render_download_buttons,
                                           render_limitations, render_section, widget_key)
from src.common.dataset_summary import DatasetSummary
from src.common.formatting import escape_text
from src.common.headers import neutralize_headers
from src.predictive import export as px_export
from src.predictive import classifiers as CLF
from src.predictive.contracts import (ALL_CLASSIFICATION_MODELS, CLASSIFICATION_MODELS, MODEL_LABELS, REGRESSION_MODELS,
                                      Settings, SettingsError, SILHOUETTE_CAVEAT)
from src.predictive.evaluate import run_supervised
from src.predictive.segment import numeric_candidates, run_segmentation
from src.predictive.task import date_like_columns, eligible_targets, target_info

MODULE_KEY = "predictive"


def _k(name: str) -> str:
    return widget_key(MODULE_KEY, f"{name}_{session_paths.get_dataset_id()}")


def _fmt(v, d=4):
    return "n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else (f"{v:,.{d}g}" if isinstance(v, (int, float)) else str(v))


def _refusal(res) -> None:
    st.warning(f"**Cannot run:** {escape_text(res.refusal.reason)}")
    if res.refusal.recommendation:
        st.caption(escape_text(res.refusal.recommendation))


def _select_all(key: str, keys: list) -> None:
    st.session_state[key] = list(keys)


def _supervised_controls(df: pd.DataFrame, summary: DatasetSummary):
    targets = eligible_targets(df)
    if not targets:
        st.warning("No column in this dataset can be used as a target (needs a numeric column with several values, or a "
                   "categorical column with 2 to 20 classes).")
        return None
    target = st.selectbox("Target (what do you want to predict?)", targets, key=_k("target"))
    info = target_info(df, target)
    task = info.task
    if info.ambiguous:
        task = st.radio("This column could be either", ["regression", "classification"], horizontal=True, key=_k("task"),
                        format_func=lambda t: "Predict a number (regression)" if t == "regression" else "Predict a category (classification)")
    st.caption(f"Detected task: **{task}**. " + (info.reason or ""))
    balance = False
    if task == "regression":
        st.info(f"**'{target}' is a continuous number, so this is a regression problem.** The 12 classifiers are not "
                "offered: classification needs categories, and cutting a continuous value into bands would invent "
                "arbitrary thresholds and throw away information. The regression models below predict the number "
                "directly.")
        models = st.multiselect("Models to compare", list(REGRESSION_MODELS), default=list(REGRESSION_MODELS),
                                key=_k(f"models_{task}"), format_func=MODEL_LABELS.get)
    else:
        avail = {k: CLF.availability(k) for k in ALL_CLASSIFICATION_MODELS}
        models = st.multiselect(
            "Models to compare (defaults: Logistic Regression, Decision Tree, Random Forest)",
            list(ALL_CLASSIFICATION_MODELS), default=list(CLASSIFICATION_MODELS), key=_k(f"models_{task}"),
            format_func=lambda k: MODEL_LABELS[k] + ("" if avail[k][0] else " (unavailable)"),
            help="Only the models you select are trained.")
        b1, b2 = st.columns([1, 1])
        b1.button("Select all available models", key=_k("all_models"), on_click=_select_all,
                  args=(_k(f"models_{task}"), [k for k in ALL_CLASSIFICATION_MODELS if avail[k][0]]))
        b2.button("Back to the defaults", key=_k("default_models"), on_click=_select_all,
                  args=(_k(f"models_{task}"), list(CLASSIFICATION_MODELS)))
        missing = [k for k in models if not avail[k][0]]
        for k in missing:
            st.warning(f"**{MODEL_LABELS[k]}** — {escape_text(avail[k][1])} It will be reported as unavailable; "
                       "no other model is used in its place.")
        vc = df[target].dropna().astype(str).value_counts()
        ratio = float(vc.max() / vc.min()) if len(vc) > 1 and vc.min() > 0 else 1.0
        balance = st.checkbox("Handle class imbalance (class weights / equal priors where the model supports them)",
                              value=ratio >= 3, key=_k(f"balance_{target}"),
                              help="On by default when the largest class is at least 3× the smallest. Models without "
                                   "a weighting option (K-Nearest Neighbors, Gradient Boosting, AdaBoost, XGBoost "
                                   "multiclass) are trained unweighted; the availability table says which.")
        st.caption(f"Class balance: largest / smallest class = {ratio:.1f}.")
        with st.expander("Classifier availability and imbalance handling"):
            st.dataframe(CLF.availability_table().drop(columns=["key"]), hide_index=True, width="stretch")
    pool = [c for c in df.columns if c != target]
    excluded = st.multiselect("Columns you do not want used as predictors", pool, key=_k("excl"))
    c1, c2, c3 = st.columns(3)
    test_share = c1.slider("Share of rows held out for the final test", 0.10, 0.40, 0.20, 0.05, key=_k("test"))
    folds = c2.slider("Cross-validation folds", 3, 10, 5, key=_k("folds"))
    seed = int(c3.number_input("Random seed", 0, 10_000_000, 42, key=_k("seed")))
    dates = date_like_columns(df)
    chrono = st.checkbox("Chronological validation (train on earlier rows, test on the latest)", key=_k("chrono"))
    tcol = None
    if chrono:
        tcol = st.selectbox("Time column", ["(current row order)"] + dates, key=_k("tcol"))
        tcol = None if tcol.startswith("(") else tcol
    simple = None
    if "simple_linear" in models:
        num = [c for c in pool if c in summary.numeric_columns and c != target]
        pick = st.selectbox("Simple Linear predictor", ["Automatic (chosen inside each fold)"] + num, key=_k("simple"))
        simple = None if pick.startswith("Automatic") else pick
    reinc = st.multiselect("Identifier-like columns you want to re-include after review", pool, key=_k("reinc"))
    skip = st.checkbox("Skip the sensitivity check (faster)", key=_k("skip"))
    try:
        stg = Settings(test_share=test_share, cv_folds=folds, seed=seed, chronological=chrono, time_col=tcol,
                       models=tuple(models), simple_predictor=simple, user_excluded=tuple(excluded),
                       reinclude_identifiers=tuple(reinc), skip_sensitivity=skip, balance_classes=balance).validate()
    except SettingsError as exc:
        st.error(str(exc))
        return None
    return target, task, stg


def _plots(res) -> None:
    c = res.comparison[res.comparison.status == "ok"]
    fig = px.bar(c, x="model", y="cv_mean", error_y="cv_sd", color="selected",
                 labels={"cv_mean": "cross-validated " + ("RMSE (lower is better)" if res.task == "regression" else "macro-F1 (higher is better)")})
    fig.update_layout(legend_title_text="Selected model")
    st.plotly_chart(apply_dark_layout(fig, "Model comparison (training portion, cross-validation)",
                                      x_title="Model", y_title=("Cross-validated RMSE (lower is better)" if res.task == "regression" else "Cross-validated macro-F1 (higher is better)")), use_container_width=True)
    if res.importance is not None and len(res.importance):
        imp = res.importance.sort_values("mean")
        fig = px.bar(imp, x="mean", y="feature", orientation="h", error_x="sd_across_folds")
        st.plotly_chart(apply_dark_layout(fig, "Permutation importance on cross-validation validation folds",
                                          x_title="Permutation importance (drop in cross-validated score)",
                                          y_title="Feature"), use_container_width=True)
    p = res.predictions
    te = p[p.data_split == "test"]
    if res.task == "regression" and len(te):
        fig = px.scatter(te, x="actual", y="predicted")
        lo, hi = float(min(te.actual.min(), te.predicted.min())), float(max(te.actual.max(), te.predicted.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction"))
        st.plotly_chart(apply_dark_layout(fig, "Actual vs predicted (held-out test rows)",
                                          x_title=f"Actual {res.target}", y_title=f"Predicted {res.target}"),
                        use_container_width=True)
    elif res.confusion is not None:
        fig = px.imshow(res.confusion, text_auto=True, aspect="auto")
        fig.update_layout(coloraxis_colorbar=dict(title=dict(text="Rows")))
        st.plotly_chart(apply_dark_layout(fig, "Confusion matrix (held-out test rows)",
                                          x_title="Predicted class", y_title="Actual class"), use_container_width=True)


CLS_METRICS = [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"), ("recall_macro", "Recall (macro)"),
               ("f1_macro", "F1 (macro)"), ("roc_auc", "ROC-AUC")]
REG_METRICS = [("r2", "R² (higher is better)"), ("mae", "MAE (lower is better)"), ("rmse", "RMSE (lower is better)"),
               ("mse", "MSE (lower is better)"), ("mape", "MAPE % (lower is better)")]


def _metric_bar(t: pd.DataFrame, col: str, label: str, title: str, y_title: str) -> None:
    d = t[t[col].notna()] if col in t else t.iloc[0:0]
    if d.empty:
        st.caption(f"{title}: not available for any model (see the notes in the table).")
        return
    fig = go.Figure(go.Bar(x=d.model, y=d[col].astype(float),
                           marker_color=["#3ddc97" if s else "#3B82F6" for s in d.selected_by_cv],
                           text=[f"{v:.3g}" for v in d[col].astype(float)], textposition="outside"))
    st.plotly_chart(apply_dark_layout(fig, title, x_title="Model", y_title=y_title), width="stretch")


def _holdout_all_section(res) -> None:
    """PART 03: every model's own evaluation on the untouched test rows. Reporting only."""
    t = res.holdout_all
    if t is None or not len(t):
        return
    render_section("Independent Holdout Evaluation of Every Model")
    st.info("Each model was refitted on the training portion and scored **once** on the same held-out test rows. "
            "These results were **not** used to choose the model — the choice above was made by cross-validation "
            "only. Green bars mark the model selected by cross-validation.")
    st.dataframe(t.round(4), hide_index=True, width="stretch")
    bad = t[t.status != "ok"]
    for r in bad.itertuples():
        st.error(f"**{escape_text(r.model)}** — {escape_text(r.status)}: {escape_text(str(r.detail))}")
    ok = t[t.status == "ok"].copy()
    if ok.empty:
        return
    if res.task == "classification":
        long = ok.melt(id_vars=["model"], value_vars=[c for c, _ in CLS_METRICS if c in ok], var_name="metric",
                       value_name="value").dropna()
        long["metric"] = long.metric.map(dict(CLS_METRICS))
        fig = px.bar(long, x="model", y="value", color="metric", barmode="group")
        fig.update_layout(legend_title_text="Metric")
        st.plotly_chart(apply_dark_layout(fig, "Model performance comparison (held-out test rows)", x_title="Model",
                                          y_title="Score (0 to 1, higher is better)"), width="stretch")
        for col, label in CLS_METRICS[:4]:
            _metric_bar(ok, col, label, f"{label} by model (held-out test rows)", f"{label} (0 to 1)")
        if res.holdout_roc:
            binary = len(res.classes) == 2
            if binary:
                fig = go.Figure()
                for m, curves in res.holdout_roc.items():
                    (name, (f, r)), = curves.items()
                    auc = ok.loc[ok.model == m, "roc_auc"]
                    fig.add_trace(go.Scatter(x=f, y=r, mode="lines",
                                             name=f"{m} (AUC {float(auc.iloc[0]):.3f})" if len(auc) and pd.notna(auc.iloc[0]) else m))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dot")))
                fig.update_layout(legend_title_text="Model")
                st.plotly_chart(apply_dark_layout(fig, f"ROC curves (held-out test rows; positive class "
                                                       f"'{res.classes[1]}')", x_title="False positive rate",
                                                  y_title="True positive rate"), width="stretch")
            else:
                m = st.selectbox("Model for the one-vs-rest ROC curves", list(res.holdout_roc), key=_k("roc_model"))
                fig = go.Figure()
                for name, (f, r) in res.holdout_roc[m].items():
                    fig.add_trace(go.Scatter(x=f, y=r, mode="lines", name=name))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dot")))
                fig.update_layout(legend_title_text="Class")
                st.plotly_chart(apply_dark_layout(fig, f"One-vs-rest ROC curves — {m} (held-out test rows)",
                                                  x_title="False positive rate", y_title="True positive rate"),
                                width="stretch")
        else:
            st.caption("ROC curves are not shown: they need class probabilities and every class present in the test rows.")
        if res.holdout_confusions:
            m = st.selectbox("Model for the confusion matrix", list(res.holdout_confusions), key=_k("cm_model"))
            fig = px.imshow(res.holdout_confusions[m], text_auto=True, aspect="auto")
            fig.update_layout(coloraxis_colorbar=dict(title=dict(text="Rows")))
            st.plotly_chart(apply_dark_layout(fig, f"Confusion matrix — {m} (held-out test rows)",
                                              x_title="Predicted class", y_title="Actual class"), width="stretch")
            with st.expander("Per-class precision, recall and F1 for this model"):
                st.dataframe(res.holdout_class_tables.get(m), hide_index=True)
    else:
        for col, label in REG_METRICS:
            _metric_bar(ok, col, label, f"{label.split(' (')[0]} by model (held-out test rows)", label)
    _metric_bar(ok, "train_seconds", "Training time", "Training time by model (full training portion)", "Seconds")
    _metric_bar(ok, "predict_seconds", "Prediction time", "Prediction time by model (all held-out rows)", "Seconds")
    if res.feature_importance_all:
        m = st.selectbox("Model for feature importance", list(res.feature_importance_all), key=_k("fi_model"))
        fi = res.feature_importance_all[m].head(20).sort_values("value")
        fig = go.Figure(go.Bar(x=fi.value, y=fi.feature, orientation="h"))
        st.plotly_chart(apply_dark_layout(fig, f"Feature importance — {m} ({fi.kind.iloc[0]})",
                                          x_title=str(fi.kind.iloc[0]).capitalize(), y_title="Feature"), width="stretch")
        st.caption("Model-internal importance describes the fitted model, not cause and effect. Support Vector Machine "
                   "(RBF), K-Nearest Neighbors and Naive Bayes have no built-in importance; see the permutation "
                   "importance above for the selected model.")


def _supervised_results(res) -> None:
    if res.status != "ok":
        _refusal(res)
        return
    render_section("Data Preparation and Validation")
    st.dataframe(pd.DataFrame({"item": list(res.data_prep), "value": [str(v) for v in res.data_prep.values()]}), hide_index=True)
    st.dataframe(pd.DataFrame(res.checks), hide_index=True)
    for w in res.warnings:
        st.caption("⚠️ " + escape_text(w))
    render_section("Model Comparison")
    st.dataframe(res.comparison, hide_index=True)
    st.success(f"**Selected: {escape_text(res.selected)}.** {escape_text(res.selection_reason)}")
    render_section("Held-out Test Result (evaluated once)")
    cols = st.columns(min(4, len(res.heldout)) or 1)
    for i, (k, v) in enumerate(res.heldout.items()):
        cols[i % len(cols)].metric(k, _fmt(v))
    if res.heldout_ci:
        st.dataframe(pd.DataFrame([{"metric": k, "lower": v[0], "upper": v[1]} for k, v in res.heldout_ci.items()]), hide_index=True)
    if res.ci_note:
        st.caption(escape_text(res.ci_note))
    if res.class_metrics is not None:
        st.dataframe(res.class_metrics, hide_index=True)
    _plots(res)
    _holdout_all_section(res)
    render_section("Leakage Review")
    if len(res.final_review):
        st.dataframe(res.final_review, hide_index=True)
    else:
        st.caption("No identifier-like, copy-of-target or suspiciously strong columns were found in the training data.")
    if len(res.sensitivity):
        st.dataframe(res.sensitivity, hide_index=True)
    with st.expander("Per-fold screening audit"):
        st.dataframe(res.fold_audit, hide_index=True)
    render_section("Feature Importance")
    st.caption(escape_text(res.internals.get("importance_note", "")))
    if res.importance is not None:
        st.dataframe(res.importance, hide_index=True)
    render_section("Downloads")
    try:
        render_download_buttons(MODULE_KEY, "predictions", table=neutralize_headers(res.predictions), sheets=px_export.supervised_sheets(res))
    except Exception as exc:  # export errors are shown, not hidden
        st.caption(f"Export unavailable: {exc}")


def _supervised_section(df, summary) -> None:
    ctl = _supervised_controls(df, summary)
    if ctl is None:
        return
    target, task, stg = ctl
    sig = (target, task, stg.fingerprint(), len(df), tuple(df.columns))
    if st.button("Run prediction", key=_k("run"), type="primary"):
        with st.spinner("Training and validating models..."):
            cache_set(MODULE_KEY, "sup", (sig, run_supervised(df, target, settings=stg, task=task)))
    cached = cache_get(MODULE_KEY, "sup")
    if cached is None:
        st.info("Choose a target and press **Run prediction**.")
        return
    if cached[0] != sig:
        st.info("Settings changed since the last run. Press **Run prediction** to refresh; the result below is from the earlier settings.")
    _supervised_results(cached[1])


def _segment_section(df, summary) -> None:
    nums = numeric_candidates(df)
    if len(nums) < 2:
        st.warning("Segmentation needs at least 2 varying numeric columns.")
        return
    feats = st.multiselect("Columns to segment on", nums, default=nums[:6], key=_k("seg_feats"))
    auto = st.checkbox("Choose the number of clusters automatically (best silhouette, k = 2 to 8)", True, key=_k("seg_auto"))
    k = None if auto else int(st.number_input("Number of clusters", 2, 12, 3, key=_k("seg_k")))
    ids = [c for c in summary.identifier_columns if c in df.columns]
    idc = st.selectbox("Identifier column to carry into the export (optional)", ["(none)"] + ids, key=_k("seg_id"))
    seed = int(st.number_input("Random seed", 0, 10_000_000, 42, key=_k("seg_seed")))
    sig = (tuple(feats), k, idc, seed, len(df), tuple(df.columns))
    if st.button("Run segmentation", key=_k("seg_run"), type="primary"):
        cache_set(MODULE_KEY, "seg", (sig, run_segmentation(df, feats, k, seed=seed, id_column=None if idc.startswith("(") else idc)))
    cached = cache_get(MODULE_KEY, "seg")
    if cached is None:
        st.info("Pick columns and press **Run segmentation**.")
        return
    if cached[0] != sig:
        st.info("Settings changed since the last run; press **Run segmentation** to refresh.")
    res = cached[1]
    if res.status != "ok":
        _refusal(res)
        return
    for w in res.warnings:
        st.caption("⚠️ " + escape_text(w))
    m = res.metrics
    c = st.columns(3)
    c[0].metric("Clusters", m["k"])
    c[1].metric("Silhouette", _fmt(m["silhouette"]))
    c[2].metric("PCA variance shown", f"{m['pca_variance_explained']:.0%}")
    st.caption(escape_text(res.caveat))
    st.dataframe(res.profiles, hide_index=True)
    st.dataframe(res.k_table, hide_index=True)
    fig = px.scatter(res.pca, x="pc1", y="pc2", color="cluster")
    fig.update_layout(legend_title_text="Cluster")
    st.plotly_chart(apply_dark_layout(fig, "Clusters (2-D projection)", x_title="Principal component 1",
                                      y_title="Principal component 2"), use_container_width=True)
    render_download_buttons(MODULE_KEY, "segments", table=neutralize_headers(res.assignments), sheets=px_export.segmentation_sheets(res))


def render_predictive_body(df: pd.DataFrame, summary: DatasetSummary) -> None:
    st.info("**Predictive Analysis** is delivered in Part 4 of Phase 8 and is implemented. It trains models only on the active dataset. Preprocessing and "
            "screening are fitted on training rows only; models are chosen by cross-validation; the test rows are used once.")
    t1, t2 = st.tabs(["Prediction (regression / classification)", "Segmentation (K-Means)"])
    with t1:
        _supervised_section(df, summary)
    with t2:
        _segment_section(df, summary)
    render_limitations([
        "A predictive model finds patterns; it does not prove cause and effect.",
        "Leakage that is not visible in the data cannot be detected automatically; judge it using business knowledge.",
        "Test results estimate performance on similar future data only if that data resembles this dataset.",
        SILHOUETTE_CAVEAT,
    ])
