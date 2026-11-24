"""Column-role review, correction and target confirmation (PART 02).

One component, used in two places:

* the Upload page (step 2) -- the person reviews the suggested business roles,
  corrects any of them and confirms the analysis target(s) before analysis;
* the four marketing-model pages -- the same review, plus saving a new mapping
  version for an uploaded dataset.

Nothing uncertain is applied silently: when several columns could be the
target, none is selected until the person chooses.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from dashboard.components import session_paths
from dashboard.components.ui import callout
from src.common.formatting import escape_text
from src.data_engine import role_engine as RE

DEMO_TARGETS = ["Revenue"]          # the bundled demo is curated: revenue is its documented target
_LABEL_TO_ROLE = {v[1]: k for k, v in RE.ROLE_INFO.items()}
_ROLE_LABELS = [RE.ROLE_INFO[r][1] for r in RE.ROLES]


def is_demo() -> bool:
    return not session_paths.is_user_dataset_mode()


def dataset_name() -> str:
    if is_demo():
        return "Demo dataset (bundled sample)"
    info = session_paths.get_active_dataset()
    return info.get("filename") or session_paths.get_dataset_id()


def _session_key() -> str:
    return f"mp_page_rolemap_{session_paths.get_dataset_id()}"


def mapping_signature(m: RE.RoleMapping) -> str:
    return json.dumps({"r": {c: [r.role, r.channel] for c, r in m.columns.items()}, "t": m.targets,
                       "v": m.version}, sort_keys=True)


def get_mapping(df: pd.DataFrame) -> RE.RoleMapping:
    """Active mapping: the session copy, else the saved file (uploads), else fresh suggestions.
    The demo starts from its documented target; uploads never get a target chosen for them."""
    key = _session_key()
    cached = st.session_state.get(key)
    if isinstance(cached, RE.RoleMapping) and cached.fingerprint == RE.fingerprint(df):
        return cached
    m = None
    if not is_demo():
        m = RE.load_mapping(session_paths.get_processed_dir(), df)
    if m is None:
        m = RE.suggest_roles(df, dataset_id=session_paths.get_dataset_id(), dataset_name=dataset_name())
        if is_demo():
            m = RE.apply_user_roles(m, {}, targets=[t for t in DEMO_TARGETS if t in m.target_candidates])
            m.source = "demo-preset"
    st.session_state[key] = m
    return m


def set_mapping(m: RE.RoleMapping) -> None:
    st.session_state[_session_key()] = m


def role_table(m: RE.RoleMapping) -> pd.DataFrame:
    rows = []
    for c, r in m.columns.items():
        rows.append({
            "Column": c,
            "Role": r.label,
            "Channel": r.channel or "",
            "Confidence": float(r.confidence),
            "Evidence": "; ".join(r.evidence),
            "Alternatives": "; ".join(f"{a['label']} ({a['confidence']:.2f})" for a in r.alternatives),
            "Warnings": " ".join(r.warnings),
        })
    return pd.DataFrame(rows)


def render_role_editor(df: pd.DataFrame, m: RE.RoleMapping, key: str) -> RE.RoleMapping:
    """Editable role table. Returns a mapping with the person's edits applied (not saved)."""
    st.caption("Each column's suggested business role, how sure the engine is, why, and what else it could be. "
               "Change the **Role** cell to correct it. Spend means advertising spend for one channel; trade or "
               "promotion spend is a separate business variable.")
    table = role_table(m)
    edited = st.data_editor(
        table, key=key, hide_index=True, width="stretch", disabled=["Column", "Confidence", "Evidence",
                                                                            "Alternatives", "Warnings"],
        column_config={
            "Role": st.column_config.SelectboxColumn("Role", options=_ROLE_LABELS, required=True, width="medium"),
            "Channel": st.column_config.TextColumn("Channel", help="Channel name for spend / exposure columns"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "Evidence": st.column_config.TextColumn("Evidence", width="large"),
        })
    changes, channels = {}, {}
    for row in edited.itertuples(index=False):
        col = row.Column
        role = _LABEL_TO_ROLE.get(row.Role)
        if role and m.columns[col].role != role:
            changes[col] = role
        if (row.Channel or "") != (m.columns[col].channel or ""):
            channels[col] = row.Channel
    if not changes and not channels:
        return m
    return RE.apply_user_roles(m, changes, channels=channels)


def render_target_choice(m: RE.RoleMapping, key: str) -> list:
    """Multiselect of outcome columns. Empty until the person chooses when the target is ambiguous."""
    cands = m.target_candidates
    if not cands:
        callout("No outcome column (revenue, units, orders, leads, profit, customers...) is mapped. Set a column's "
                "role to an outcome in the table above to analyse it.", kind="warn", label="Target needed")
        return []
    default = [t for t in m.targets if t in cands]
    if not default and len(cands) == 1 and m.columns[cands[0]].confidence >= 0.75:
        default = list(cands)           # a single, clearly named outcome is not ambiguous
    if len(cands) > 1 and not m.targets:
        callout("Several columns could be the analysis target: " + ", ".join(f"**{c}**" for c in cands)
                + ". Choose one or more; none is selected for you. Each target is analysed separately.",
                kind="warn", label="Please confirm the target")
    chosen = st.multiselect("Analysis target(s)", cands, default=default, key=key,
                            help="Each chosen outcome can be analysed on its own on the model pages.")
    return chosen


def mode_banner(mode: RE.ModeResult) -> None:
    kind = {"spend": "good", "exposure": "info", "general": "warn"}[mode.mode]
    text = f"**{mode.label}** — " + "; ".join(mode.evidence) + "."
    callout(text, kind=kind, label="Detected analytics mode")


def render_upload_roles(df: pd.DataFrame, file_hash: str, name: str) -> tuple[RE.RoleMapping, bool]:
    """Upload page step: suggestions -> edits -> target confirmation. Returns (mapping, ready)."""
    m0 = _upload_suggestions(file_hash, df, name)
    st.caption("Roles are suggested from column names, values, statistics and relationships between columns. "
               "Review them; nothing uncertain is applied without your confirmation.")
    m = render_role_editor(df, m0, key=f"mp_upload_roles_{file_hash[:12]}")
    targets = render_target_choice(m, key=f"mp_upload_targets_{file_hash[:12]}")
    m = RE.apply_user_roles(m, {}, targets=targets)
    mode_banner(RE.detect_mode(m))
    for w in m.warnings:
        st.caption("• " + escape_text(w))
    ready = bool(m.targets) or not m.target_candidates
    if m.target_candidates and not m.targets:
        st.caption("Choose the analysis target(s) above to continue.")
    return m, ready


@st.cache_data(show_spinner=False, max_entries=8)
def _upload_suggestions_cached(file_hash: str, _df: pd.DataFrame, name: str) -> dict:
    return RE.suggest_roles(_df, dataset_name=name).to_dict()


def _upload_suggestions(file_hash: str, df: pd.DataFrame, name: str) -> RE.RoleMapping:
    return RE.RoleMapping.from_dict(_upload_suggestions_cached(file_hash, df, name))


def save_upload_mapping(dataset_id: str, m: RE.RoleMapping, name: str) -> RE.RoleMapping:
    """Persist the confirmed upload mapping as version 1 against the stored analysis table
    (cleaning can change the row count, so the fingerprint is taken from what the pages will read)."""
    from config.settings import dataset_processed_dir
    out = RE.apply_user_roles(m, {}, targets=m.targets)
    out.dataset_id, out.dataset_name = dataset_id, name
    for c in out.columns.values():
        c.user_set = True
    d = dataset_processed_dir(dataset_id)
    table = d / "standardized_data.csv"
    if table.exists():
        stored = pd.read_csv(table, nrows=None)
        missing = [c for c in out.columns if c not in stored.columns]
        if missing:  # keep only columns the stored table has
            for c in missing:
                out.columns.pop(c, None)
        out.fingerprint = RE.fingerprint(stored)
        out.n_rows = len(stored)
    return RE.save_mapping(d, out, note="confirmed on upload")


def render_mapping_panel(df: pd.DataFrame, m: RE.RoleMapping, page_key: str, *, expanded: bool) -> RE.RoleMapping:
    """Model-page panel: review / correct roles and targets; uploads can save a new mapping version."""
    with st.expander(f"Column roles and target · mapping version {m.version or '—'}"
                     + (" · not yet confirmed" if not m.confirmed and m.source != "demo-preset" else ""),
                     expanded=expanded):
        if m.source == "demo-preset":
            st.caption("Demo dataset: roles are suggested by the engine and Revenue is the documented target. Changes "
                       "here apply to this session only.")
        edited = render_role_editor(df, m, key=f"mp_page_roles_{page_key}_{session_paths.get_dataset_id()}")
        targets = render_target_choice(edited, key=f"mp_page_targets_{page_key}_{session_paths.get_dataset_id()}")
        edited = RE.apply_user_roles(edited, {}, targets=targets)
        changed = mapping_signature(edited) != mapping_signature(m)
        c1, c2 = st.columns([1, 3])
        if is_demo():
            if c1.button("Apply for this session", key=f"mp_page_apply_{page_key}", disabled=not changed):
                set_mapping(edited)
                st.rerun()
        else:
            if c1.button("Save mapping", key=f"mp_page_save_{page_key}", type="primary",
                         disabled=not (changed or not m.confirmed) or (bool(edited.target_candidates) and not edited.targets)):
                saved = RE.save_mapping(session_paths.get_processed_dir(), edited, note=f"saved on {page_key}")
                set_mapping(saved)
                st.rerun()
        if changed:
            c2.caption("You have unsaved changes; results below still use the current mapping.")
    return m
