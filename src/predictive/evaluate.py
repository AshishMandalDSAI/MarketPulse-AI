"""Supervised workflow P0 -> P4 (see package docstring). No Streamlit; deterministic given the settings."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split

from src.predictive import importance as imp
from src.predictive import leakage
from src.predictive import metrics as M
from src.predictive.contracts import (BOOTSTRAP_NOTE, BOOTSTRAP_NOTE_CHRONO, ALL_CLASSIFICATION_MODELS, CLASSIFICATION_MODELS, IMPORTANCE_NOTE,
                                      LEAKAGE_LIMIT, MAX_CLASSES, MIN_CHRONO_BLOCK, MIN_CHRONO_FOLDS, MIN_CLASS_ROWS,
                                      MIN_LABELLED, MIN_TEST_FOR_CI, MIN_TEST_WARN, MODEL_LABELS, REGRESSION_MODELS,
                                      ROW_CAP, SENSITIVITY_MAX, STRONG_WARN_LABELLED, WARN_CLASS_ROWS, WARN_LABELLED,
                                      Settings, SupervisedResult)
from src.predictive.duplicates import handle_duplicates
from src.predictive.classifiers import ModelUnavailable, roc_points, safe_log_loss, weighted_from_cm
from src.predictive.procedure import ModelNotApplicable, ScreenedModel
from src.predictive.task import candidate_features, target_info

LOWER_IS_BETTER = {"regression": True, "classification": False}


def _check(res, name, status, detail):
    res.checks.append({"check": name, "status": status, "detail": detail})


def _clean_target(s: pd.Series, task: str) -> pd.Series:
    if task == "regression":
        return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    return s.where(s.isna(), s.astype(str))


def _cap_indices(y: pd.Series, task: str, seed: int, chrono: bool) -> np.ndarray:
    n = len(y)
    pos = np.arange(n)
    if n <= ROW_CAP:
        return pos
    if chrono:
        return pos[-ROW_CAP:]                      # contiguous most recent window; order preserved
    rng = np.random.default_rng(seed)
    if task == "regression":
        return np.sort(rng.choice(pos, ROW_CAP, replace=False))
    counts = y.value_counts()
    floor = {c: min(int(v), 20) for c, v in counts.items()}
    rest = ROW_CAP - sum(floor.values())
    take = {}
    total = float(sum(counts.values()))
    for c, v in counts.items():
        take[c] = min(int(v), floor[c] + int(round(max(rest, 0) * v / total)))
    idx = []
    for c, k in take.items():
        members = pos[(y == c).to_numpy()]
        idx.append(rng.choice(members, k, replace=False))
    sel = np.sort(np.concatenate(idx))
    if len(sel) > ROW_CAP:
        sel = np.sort(rng.choice(sel, ROW_CAP, replace=False))
    return sel


def _folds(n_train, y_train, task, k, chrono, seed):
    """Return (list of (tr, va) position arrays, k_used, note)."""
    pos = np.arange(n_train)
    if chrono:
        kk = k
        while kk >= MIN_CHRONO_FOLDS:
            splits = list(TimeSeriesSplit(n_splits=kk).split(pos))
            if min(len(tr) for tr, _ in splits) >= MIN_CHRONO_BLOCK:
                return splits, kk, (f"cross-validation folds reduced from {k} to {kk} so every training block has "
                                    f"at least {MIN_CHRONO_BLOCK} rows" if kk != k else "")
            kk -= 1
        return None, 0, (f"time-ordered cross-validation needs training blocks of at least {MIN_CHRONO_BLOCK} rows "
                         f"with at least {MIN_CHRONO_FOLDS} folds; the training portion is too short")
    if task == "classification":
        kk = min(k, int(pd.Series(y_train).value_counts().min()))
        if kk < 2:
            return None, 0, "a class has fewer than 2 training rows, so stratified cross-validation is impossible"
        splits = list(StratifiedKFold(kk, shuffle=True, random_state=seed).split(pos, y_train))
    else:
        kk = min(k, n_train)
        splits = list(KFold(kk, shuffle=True, random_state=seed).split(pos))
    return splits, kk, (f"cross-validation folds reduced from {k} to {kk} (small training portion)" if kk != k else "")


def _score(task, y_true, y_pred, classes):
    return M.selection_score(task, y_true, y_pred, classes)


def _cv_model(key, task, cands, X_tr, y_tr, splits, fold_screens, classes, st, forced_out=(), reinclude=None,
              keep_preds=False):
    """Cross-validate one model; screening and fitting happen inside each fold on T_f only."""
    reinclude = set(st.reinclude_identifiers) if reinclude is None else reinclude
    val_scores, train_scores, preds, t0 = [], [], {}, time.perf_counter()
    for f, (tr, va) in enumerate(splits):
        Xt, yt, Xv, yv = X_tr.iloc[tr], y_tr.iloc[tr], X_tr.iloc[va], y_tr.iloc[va]
        m = ScreenedModel(key, task, cands, st.seed, classes, st.user_excluded, reinclude, st.simple_predictor,
                          forced_out, partition=f"T_{f + 1}", screen=fold_screens.get(f),
                          balanced=bool(getattr(st, "balance_classes", False))).fit(Xt, yt)
        pv, prob = m.predict_all(Xv)
        val_scores.append(_score(task, yv, pv, classes))
        train_scores.append(_score(task, yt, m.predict(Xt), classes))
        if keep_preds:
            preds[f] = (va, pv, prob)
    return val_scores, train_scores, preds, time.perf_counter() - t0


def run_supervised(df: pd.DataFrame, target: str, features: Optional[list] = None, settings: Optional[Settings] = None,
                   task: Optional[str] = None) -> SupervisedResult:
    st = (settings or Settings()).validate()
    res = SupervisedResult(settings=st, target=target)
    if df is None or len(df) == 0 or target not in df.columns:
        return res.refuse("The target column was not found in the active dataset.", "Choose a target from the list.")
    info = target_info(df, target)
    if not info.eligible:
        return res.refuse(f"'{target}' cannot be used as a target: {info.reason}", "Choose a different target column.")
    task = task or info.task
    if task != info.task and not info.ambiguous:
        return res.refuse(f"'{target}' can only be treated as a {info.task} target.", "Use the detected task.")
    res.task = task
    res.selection_metric = "cross-validated RMSE (lower is better)" if task == "regression" \
        else "cross-validated macro-F1 (higher is better)"
    # PART 03: all 12 classifiers are allowed; with no explicit choice the defaults stay the original three.
    allowed = REGRESSION_MODELS if task == "regression" else ALL_CLASSIFICATION_MODELS
    default = REGRESSION_MODELS if task == "regression" else CLASSIFICATION_MODELS
    models = [m for m in (st.models or default) if m in allowed]
    if not models:
        return res.refuse("No model was selected.", "Tick at least one model.")

    # ----------------------------------------------------------------- P0
    time_col = st.time_col if (st.chronological and st.time_col) else None
    cands, excluded_dtype = candidate_features(df, target, time_col)
    if features is not None:
        cands = [c for c in cands if c in features]
    if time_col and time_col not in df.columns:
        return res.refuse(f"The time column '{time_col}' is not in the dataset.", "Choose another time column.")
    if not cands:
        return res.refuse("No candidate predictor columns are available.",
                          "The dataset needs at least one non-date column besides the target.")
    work = df[[target] + cands + ([time_col] if time_col else [])].copy()
    work.index = np.arange(len(df))                      # row id = position in the active table
    prep = {"rows_in_table": int(len(df)), "invalid_time_rows_excluded": 0}
    # +/- infinity in a numeric feature cannot be modelled; treat it as missing (a target-independent, row-wise rule
    # applied before the split, so it cannot leak) and say so.
    n_inf = 0
    for c in cands:
        if pdt.is_numeric_dtype(work[c]) and not pdt.is_bool_dtype(work[c]):
            m = np.isinf(pd.to_numeric(work[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan))
            if m.any():
                n_inf += int(m.sum())
                work[c] = work[c].astype(float).mask(m)
    prep["infinite_feature_values_treated_as_missing"] = n_inf
    if n_inf:
        res.warnings.append(f"{n_inf} infinite feature value(s) were treated as missing and filled using training rows only.")
    if st.chronological and time_col:
        t = pd.to_datetime(work[time_col], errors="coerce")
        bad = t.isna()
        prep["invalid_time_rows_excluded"] = int(bad.sum())
        work = work.loc[~bad].assign(**{time_col: t[~bad]}).sort_values(time_col, kind="mergesort")
    y_all = _clean_target(work[target], task)
    work[target] = y_all
    unlabelled_mask = y_all.isna()
    prep["unlabelled_rows_missing_or_invalid_target"] = int(unlabelled_mask.sum())
    unlabelled = work.loc[unlabelled_mask]
    lab = work.loc[~unlabelled_mask]
    feats_for_dups = [c for c in cands if c not in st.user_excluded]
    lab, dup_report, conflicts = handle_duplicates(lab, target, feats_for_dups, time_col)
    prep.update(dup_report)
    res.conflicts = conflicts
    prep["labelled_rows_after_duplicates"] = int(len(lab))
    if dup_report["conflicting_label_rows"]:
        res.warnings.append(f"{dup_report['conflicting_label_rows']} rows share identical features but have different "
                            "targets; they are kept and reported, and they limit the accuracy any model can reach.")
    if len(lab) < MIN_LABELLED:
        res.data_prep = prep
        return res.refuse(f"Only {len(lab)} usable labelled rows remain (at least {MIN_LABELLED} are required for an "
                          "honest train/test evaluation).", "Add more rows or choose another target.")
    classes = []
    if task == "classification":
        vc = lab[target].value_counts()
        classes = sorted(vc.index.tolist())
        if len(classes) < 2:
            res.data_prep = prep
            return res.refuse("The target has a single class among the usable rows.", "Choose another target.")
        if len(classes) > MAX_CLASSES:
            res.data_prep = prep
            return res.refuse(f"The target has {len(classes)} classes (limit {MAX_CLASSES}).", "Choose another target.")
        if int(vc.min()) < MIN_CLASS_ROWS:
            res.data_prep = prep
            return res.refuse(f"Class '{vc.idxmin()}' has only {int(vc.min())} rows (at least {MIN_CLASS_ROWS} needed).",
                              "Merge or remove very rare classes, or choose another target.")
        if int(vc.min()) < WARN_CLASS_ROWS:
            res.warnings.append(f"Class '{vc.idxmin()}' has only {int(vc.min())} rows; its metrics are unreliable.")
        ratio = vc.max() / vc.min()
        if ratio >= 3:
            res.warnings.append(f"Classes are imbalanced (largest/smallest = {ratio:.1f}); macro-F1 is used for selection "
                                "so every class counts equally.")
    res.classes = classes
    if len(lab) < STRONG_WARN_LABELLED:
        res.warnings.append(f"Only {len(lab)} labelled rows: results are indicative only.")
    elif len(lab) < WARN_LABELLED:
        res.warnings.append(f"Only {len(lab)} labelled rows: expect wide uncertainty.")

    keep = _cap_indices(lab[target].reset_index(drop=True), task, st.seed, bool(st.chronological))
    capped_out = lab.iloc[np.setdiff1d(np.arange(len(lab)), keep)]
    lab = lab.iloc[keep]
    prep["rows_capped_out"] = int(len(capped_out))
    prep["row_cap"] = ROW_CAP
    prep["sampling_method"] = ("none" if not len(capped_out) else
                               "most recent contiguous window (time order preserved)" if st.chronological else
                               "seeded stratified sample (every class kept)" if task == "classification" else "seeded random sample")
    prep["labelled_rows_analysed"] = int(len(lab))

    n_test = max(1, int(round(st.test_share * len(lab))))
    if st.chronological:
        tr_pos, te_pos = np.arange(len(lab) - n_test), np.arange(len(lab) - n_test, len(lab))
    else:
        try:
            tr_pos, te_pos = train_test_split(np.arange(len(lab)), test_size=n_test, random_state=st.seed,
                                              stratify=lab[target].to_numpy() if task == "classification" else None)
        except ValueError as exc:
            res.data_prep = prep
            return res.refuse(f"The test portion is too small to keep every class represented ({exc}).",
                              "Increase the test share or use more rows.")
        tr_pos, te_pos = np.sort(tr_pos), np.sort(te_pos)
    T, TE = lab.iloc[tr_pos], lab.iloc[te_pos]
    X_tr, y_tr = T[cands], T[target]
    X_te, y_te = TE[cands], TE[target]
    prep.update({"train_rows": int(len(T)), "test_rows": int(len(TE)), "seed": st.seed,
                 "split": "chronological (last rows = test)" if st.chronological else "random" +
                 (", stratified" if task == "classification" else ""),
                 "time_column": time_col or ("current row order" if st.chronological else "")})
    if task == "classification" and y_tr.nunique() < 2:
        res.data_prep = prep
        return res.refuse("The training portion contains a single class.", "Use a random split or more data.")
    if len(TE) < MIN_TEST_WARN:
        res.warnings.append(f"The test set has only {len(TE)} rows; the held-out result is very uncertain.")
    if st.chronological and not time_col:
        res.warnings.append("No time column was chosen: the current row order is assumed to be chronological.")
    if st.chronological:
        res.warnings.append("Chronological validation checks a supervised model on later rows; it is not forecasting. "
                            "A target built from future values can still leak; judge this with business knowledge.")
    res.data_prep = prep
    _check(res, "Enough labelled rows", "pass", f"{len(lab)} labelled rows analysed (minimum {MIN_LABELLED}).")
    _check(res, "Target usable", "pass", f"'{target}' treated as a {task} target.")
    _check(res, "Duplicates", "warn" if dup_report["conflicting_label_rows"] else "pass",
           f"{dup_report['exact_duplicate_rows_removed']} exact duplicate rows removed; "
           f"{dup_report['conflicting_label_rows']} rows with conflicting labels kept.")
    _check(res, "Held-out test set", "warn" if len(TE) < MIN_TEST_WARN else "pass", f"{len(TE)} test rows reserved.")

    splits, k_used, note = _folds(len(T), y_tr.to_numpy(), task, st.cv_folds, bool(st.chronological), st.seed)
    if splits is None:
        return res.refuse(note.capitalize() + ".", "Use more rows, a smaller test share or random validation.")
    if note:
        res.warnings.append(note[0].upper() + note[1:] + ".")
    res.cv_folds_used = k_used
    _check(res, "Cross-validation", "pass", f"{k_used} {'time-ordered ' if st.chronological else ''}folds on the training portion.")

    # ------------------------------------------------- P1 per-fold screens (T_f only)
    fold_screens = {}
    audit = []
    for f, (tr, va) in enumerate(splits):
        Xt, yt = X_tr.iloc[tr], y_tr.iloc[tr]
        scr = leakage.screen_features(Xt, yt, task, f"T_{f + 1}", st.seed)
        fold_screens[f] = scr
        audit.append({"fold": f + 1, "partition": scr.partition, "rows_screened": scr.n_rows,
                      "first_row_id": min(scr.row_ids) + 1, "last_row_id": max(scr.row_ids) + 1,
                      "eligible_features": len(scr.eligible),
                      "excluded": "; ".join(f"{c} ({v['tier']})" for c, v in scr.excluded.items()),
                      "tier2_flagged": "; ".join(scr.tier2)})
    res.fold_audit = pd.DataFrame(audit)

    # ------------------------------------------------- P1/P2 cross-validation (selection by CV only)
    rows, cv_scores, oof = [], {}, {}
    for key in list(models) + ["baseline"]:
        label = MODEL_LABELS[key]
        try:
            vs, ts, preds, secs = _cv_model(key, task, cands, X_tr, y_tr, splits, fold_screens, classes, st,
                                            keep_preds=True)
            cv_scores[key] = vs
            oof[key] = preds
            rows.append({"model": label, "key": key, "status": "ok", "cv_mean": float(np.mean(vs)),
                         "cv_sd": float(np.std(vs, ddof=1)) if len(vs) > 1 else float("nan"),
                         "in_fold_train_mean": float(np.mean(ts)), "fit_seconds": round(secs, 2), "detail": ""})
        except ModelNotApplicable as exc:
            rows.append({"model": label, "key": key, "status": "not applicable", "cv_mean": np.nan, "cv_sd": np.nan,
                         "in_fold_train_mean": np.nan, "fit_seconds": 0.0, "detail": str(exc)})
        except ModelUnavailable as exc:                            # optional package missing: never substituted
            rows.append({"model": label, "key": key, "status": "unavailable", "cv_mean": np.nan, "cv_sd": np.nan,
                         "in_fold_train_mean": np.nan, "fit_seconds": 0.0, "detail": str(exc)})
        except Exception as exc:                                   # recorded, never swallowed
            rows.append({"model": label, "key": key, "status": "failed", "cv_mean": np.nan, "cv_sd": np.nan,
                         "in_fold_train_mean": np.nan, "fit_seconds": 0.0, "detail": f"{type(exc).__name__}: {exc}"})
    comp = pd.DataFrame(rows)
    ok = comp[(comp.status == "ok") & (comp.key != "baseline")]
    if ok.empty:
        res.comparison = comp.drop(columns="key")
        return res.refuse("No model could be fitted in every cross-validation fold: "
                          + "; ".join(f"{r.model}: {r.detail}" for r in comp.itertuples() if r.detail),
                          "Use more rows or fewer restrictions.")
    order = {k: i for i, k in enumerate(models)}
    best = None
    for r in ok.itertuples():
        if best is None:
            best = r
            continue
        better = (r.cv_mean < best.cv_mean - 1e-12) if task == "regression" else (r.cv_mean > best.cv_mean + 1e-12)
        tie = abs(r.cv_mean - best.cv_mean) <= 1e-12 and order[r.key] < order[best.key]
        if better or tie:
            best = r
    sel = best.key
    base_row = comp[comp.key == "baseline"].iloc[0]
    beaten = None
    if base_row.status == "ok":
        beaten = bool(best.cv_mean < base_row.cv_mean) if task == "regression" else bool(best.cv_mean > base_row.cv_mean)
    res.baseline_beaten = beaten
    if beaten is False:
        res.warnings.append("The selected model does not beat the baseline in cross-validation; its predictions add "
                            "little beyond predicting the average / most frequent class.")
    comp["selected"] = comp.key == sel
    res.comparison = comp.drop(columns="key").reset_index(drop=True)
    res.selected = MODEL_LABELS[sel]
    res.internals["selected_key"] = sel
    res.selection_reason = (f"{MODEL_LABELS[sel]} has the {'lowest' if task == 'regression' else 'highest'} mean "
                            f"cross-validated {'RMSE' if task == 'regression' else 'macro-F1'} "
                            f"({best.cv_mean:.4g}) on the training portion. Test rows played no part in the choice.")

    # ------------------------------------------------- P3 final full-training review and refit (T only)
    scr_T = leakage.screen_features(X_tr, y_tr, task, "T", st.seed)
    try:
        final = ScreenedModel(sel, task, cands, st.seed, classes, st.user_excluded, st.reinclude_identifiers,
                              st.simple_predictor, partition="T", screen=scr_T,
                              balanced=bool(getattr(st, "balance_classes", False))).fit(X_tr, y_tr)
    except ModelNotApplicable as exc:
        return res.refuse(f"The selected model could not be refitted on the full training portion: {exc}",
                          "Adjust the feature selection.")
    res.internals["final_screen"] = scr_T
    res.internals["fold_screens"] = fold_screens
    review = []
    flagged = {}
    for c in cands:
        e = scr_T.excluded.get(c)
        in_final = c in final.cols_
        k_ex = sum(1 for s in fold_screens.values() if c in s.excluded)
        tier, reason, ev = "", "", {}
        if c in st.user_excluded:
            tier, reason = "user", "excluded by you"
        elif e:
            tier, reason, ev = e["tier"], e["reason"], e.get("evidence", {})
        elif c in scr_T.tier2:
            tier, reason, ev = "2", scr_T.tier2[c]["reason"], scr_T.tier2[c]
        elif c in scr_T.eligible and not in_final:
            tier, reason = "model", "not used by the selected model"
        if tier in ("", "model") and k_ex == 0 and c not in scr_T.tier2:
            continue
        decision = ("included after your review" if in_final and tier == "1b"
                    else "excluded" if not in_final else "kept (warning)" if tier == "2" else "included")
        review.append({"column": c, "tier": tier, "decision": decision, "reason": reason,
                       "evidence": "; ".join(f"{k}={v}" for k, v in ev.items() if k not in ("reason",)),
                       "excluded_in_cv_folds": f"{k_ex} of {len(fold_screens)}", "partition": "T"})
        if tier in ("1b", "2"):
            flagged[c] = tier
    res.final_review = pd.DataFrame(review, columns=["column", "tier", "decision", "reason", "evidence",
                                                     "excluded_in_cv_folds", "partition"])
    for c, e in scr_T.tier2.items():
        if "kept" in e.get("reason", ""):
            pass

    # sensitivity (informational; never changes any screen)
    sens = []
    if not st.skip_sensitivity and flagged:
        cur = float(best.cv_mean)
        for c, tier in sorted(flagged.items(), key=lambda kv: (kv[1], kv[0]))[:SENSITIVITY_MAX]:
            try:
                if tier == "1b":
                    vs, *_ = _cv_model(sel, task, cands, X_tr, y_tr, splits, fold_screens, classes, st,
                                       reinclude=set(st.reinclude_identifiers) | {c})
                    variant = "with the column included"
                else:
                    vs, *_ = _cv_model(sel, task, cands, X_tr, y_tr, splits, fold_screens, classes, st, forced_out={c})
                    variant = "without the column"
                sens.append({"column": c, "tier": tier, "variant": variant, "cv_score_current": cur,
                             "cv_score_variant": float(np.mean(vs)), "delta": float(np.mean(vs)) - cur,
                             "note": "informational only; no screening decision was changed"})
            except ModelNotApplicable as exc:
                sens.append({"column": c, "tier": tier, "variant": "n/a", "cv_score_current": cur,
                             "cv_score_variant": np.nan, "delta": np.nan, "note": str(exc)})
        if len(flagged) > SENSITIVITY_MAX:
            res.warnings.append(f"Sensitivity was checked for {SENSITIVITY_MAX} of {len(flagged)} flagged columns.")
    res.sensitivity = pd.DataFrame(sens, columns=["column", "tier", "variant", "cv_score_current", "cv_score_variant",
                                                  "delta", "note"])
    if any(v == "2" for v in flagged.values()):
        res.warnings.append("Some features are suspiciously strong. They are kept by default; review the Leakage Review "
                            "table and exclude any that would not be known when a real prediction is needed.")

    # importance on CV validation folds (selected model only; test rows not involved)
    rows_i = []
    for f, (tr, va) in enumerate(splits):
        m = ScreenedModel(sel, task, cands, st.seed, classes, st.user_excluded, st.reinclude_identifiers,
                          st.simple_predictor, partition=f"T_{f + 1}", screen=fold_screens[f],
                          balanced=bool(getattr(st, "balance_classes", False))).fit(X_tr.iloc[tr], y_tr.iloc[tr])
        rows_i += imp.fold_importance(m, X_tr.iloc[va], y_tr.iloc[va], task, classes, st.seed, f + 1)
    res.importance_folds = pd.DataFrame(rows_i, columns=["feature", "fold", "importance"])
    res.importance = imp.summarise(rows_i, len(splits))
    res.internal_importance = final.internal_importance()
    res.internals["importance_note"] = IMPORTANCE_NOTE

    # ------------------------------------------------- P4 one-time held-out evaluation
    _t_pred = time.perf_counter()
    pred_te, proba_te = final.predict_all(X_te)
    res.internals["selected_predict_seconds"] = time.perf_counter() - _t_pred
    if task == "regression":
        hm = M.regression_metrics(y_te, pred_te)
        res.heldout = hm
        if hm["mape"] is None:
            res.warnings.append("MAPE is undefined because some actual test values are zero.")
    else:
        hm = M.classification_metrics(y_te, pred_te, classes, proba_te)
        res.heldout = {k: v for k, v in hm.items() if not k.startswith("_")}
        res.class_metrics = hm["_table"]
        res.confusion = pd.DataFrame(hm["_cm"], index=[f"actual {c}" for c in classes],
                                     columns=[f"predicted {c}" for c in classes])
        if res.heldout.get("roc_auc") is None:
            res.warnings.append("ROC AUC is undefined for this test set (a class is missing or only one class is present).")
        if res.heldout.get("n_unseen_labels"):
            res.warnings.append("Some test labels never occurred in training; they are counted as errors.")
    if len(TE) >= MIN_TEST_FOR_CI:
        ci, notes = M.bootstrap_ci(task, y_te.to_numpy(), pred_te, classes, proba_te, st.seed)
        res.heldout_ci = ci
        res.ci_note = BOOTSTRAP_NOTE + (" " + BOOTSTRAP_NOTE_CHRONO if st.chronological else "")
        res.warnings.extend(notes)
    else:
        res.ci_note = f"Confidence intervals are omitted because the test set has fewer than {MIN_TEST_FOR_CI} rows."

    # ------------------------------------------------- P5 (PART 03) independent holdout evaluation of every model
    # Reporting only: the choice above was made by cross-validation and is not revisited. Each model is refitted on
    # the full training portion (its own screening and preprocessing on T only) and scored once on the same untouched
    # test rows, with fit and prediction times.
    _holdout_all(res, task, comp, sel, final, pred_te, proba_te, cands, X_tr, y_tr, X_te, y_te, classes, st, scr_T)

    # ------------------------------------------------- predictions export table
    res.predictions = _predictions(task, classes, T, TE, unlabelled, capped_out, splits, oof.get(sel, {}), final,
                                   pred_te, proba_te, cands, target)
    res.status = "ok"
    res.warnings.append(LEAKAGE_LIMIT)
    return res


def _predictions(task, classes, T, TE, unl, capped, splits, oof_sel, final, pred_te, proba_te, cands, target):
    def block(frame, split, basis, pred, proba, actual=True):
        d = {"source_row": frame.index.to_numpy() + 1, "data_split": split, "prediction_basis": basis,
             "actual": frame[target].to_numpy() if actual else np.nan, "predicted": pred}
        out = pd.DataFrame(d)
        if task == "regression":
            out["error"] = out["predicted"].astype(float) - out["actual"].astype(float)
            out["abs_error"] = out["error"].abs()
        else:
            out["correct"] = np.where(out["actual"].isna(), np.nan, (out["actual"].astype(str) == out["predicted"].astype(str)).astype(float))
            for j, c in enumerate(classes):
                out[f"probability_{c}"] = proba[:, j] if proba is not None else np.nan
        return out
    # out-of-fold rows
    n = len(T)
    pred_oof = np.full(n, np.nan, dtype=object); prob_oof = np.full((n, len(classes)), np.nan)
    for f, (va, pv, prob) in oof_sel.items():
        pred_oof[va] = pv
        if prob is not None:
            prob_oof[va] = prob
    parts = [block(T, "train", "out_of_fold", pred_oof if task == "classification" else pred_oof.astype(float),
                   prob_oof if task == "classification" else None)]
    parts[0].loc[parts[0]["predicted"].isna(), "prediction_basis"] = ""
    parts.append(block(TE, "test", "held_out", pred_te, proba_te))
    for frame, name in ((unl, "unlabelled"), (capped, "capped_out")):
        if len(frame):
            p, pr = final.predict_all(frame[cands])
            parts.append(block(frame, name, "final_model_scored", p, pr, actual=(name == "capped_out")))
    out = pd.concat(parts, ignore_index=True).sort_values("source_row", kind="mergesort").reset_index(drop=True)
    return out


def _holdout_all(res, task, comp, sel, final, pred_sel, proba_sel, cands, X_tr, y_tr, X_te, y_te, classes, st, scr_T):
    """One holdout row per model that passed cross-validation (PART 03). Never used for selection."""
    rows = []
    label_of = dict(zip(comp.key, comp.model)) if "key" in comp else {}
    ok_keys = [r.key for r in comp.itertuples() if r.status == "ok"]
    balanced = bool(getattr(st, "balance_classes", False))
    for key in ok_keys:
        label = MODEL_LABELS[key]
        try:
            t0 = time.perf_counter()
            if key == sel:
                # The selected model's single held-out evaluation (P4) is reused, not repeated. Its training time is
                # measured with a fresh fit on the training rows only (no test rows involved).
                m = final
                ScreenedModel(key, task, cands, st.seed, classes, st.user_excluded, st.reinclude_identifiers,
                              st.simple_predictor, partition="T", screen=scr_T, balanced=balanced).fit(X_tr, y_tr)
                fit_s = time.perf_counter() - t0
                pv, pr, pred_s = pred_sel, proba_sel, res.internals.get("selected_predict_seconds", float("nan"))
            else:
                m = ScreenedModel(key, task, cands, st.seed, classes, st.user_excluded, st.reinclude_identifiers,
                                  st.simple_predictor, partition="T", screen=scr_T, balanced=balanced).fit(X_tr, y_tr)
                fit_s = time.perf_counter() - t0
                t1 = time.perf_counter()
                pv, pr = m.predict_all(X_te)
                pred_s = time.perf_counter() - t1
            row = {"model": label, "selected_by_cv": key == sel, "status": "ok", "train_seconds": fit_s,
                   "predict_seconds": pred_s, "test_rows": int(len(y_te)), "detail": ""}
            if task == "regression":
                hm = dict(res.heldout) if key == sel else M.regression_metrics(y_te, pv)
                row.update(r2=hm["r2"], mae=hm["mae"], mse=hm["rmse"] ** 2, rmse=hm["rmse"], mape=hm["mape"])
            else:
                if key == sel:
                    hm = dict(res.heldout, _cm=res.confusion.to_numpy(), _table=res.class_metrics)
                else:
                    hm = M.classification_metrics(y_te, pv, classes, pr)
                row.update(accuracy=hm["accuracy"], balanced_accuracy=hm["balanced_accuracy"],
                           precision_macro=hm["macro_precision"], recall_macro=hm["macro_recall"],
                           f1_macro=hm["macro_f1"], **{k.replace("weighted_", "") + "_weighted": v
                                                       for k, v in weighted_from_cm(hm["_cm"]).items()},
                           roc_auc=hm["roc_auc"], log_loss=safe_log_loss(y_te, pr, classes),
                           probabilities="yes" if pr is not None else "no")
                res.holdout_confusions[label] = pd.DataFrame(hm["_cm"], index=[f"actual {c}" for c in classes],
                                                             columns=[f"predicted {c}" for c in classes])
                res.holdout_class_tables[label] = hm["_table"]
                curves = roc_points(y_te, pr, classes)
                if curves:
                    res.holdout_roc[label] = curves
            imp_tbl = m.internal_importance()
            if len(imp_tbl):
                res.feature_importance_all[label] = imp_tbl
        except Exception as exc:                                   # reported per model, never hidden
            row = {"model": label, "selected_by_cv": key == sel, "status": "failed", "train_seconds": np.nan,
                   "predict_seconds": np.nan, "test_rows": int(len(y_te)), "detail": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
    for r in comp.itertuples():
        if r.status != "ok" and r.key != "baseline":
            rows.append({"model": MODEL_LABELS[r.key], "selected_by_cv": False, "status": r.status,
                         "train_seconds": np.nan, "predict_seconds": np.nan, "test_rows": int(len(y_te)),
                         "detail": r.detail})
    res.holdout_all = pd.DataFrame(rows)
