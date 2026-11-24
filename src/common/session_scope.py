"""Dataset-scoped session storage (pure logic, works on any mapping).

Every value a Phase 8 page keeps between Streamlit reruns is stored together
with the id of the dataset it was computed from, under a key that starts with
``mp_page_``. That prefix is what ``session_paths._clear_page_caches`` already
wipes on every dataset switch/reset, and the stored dataset id is a second,
independent guard: a value computed for dataset A is never returned while
dataset B is active, even if a clear were somehow missed.
"""

from __future__ import annotations

from typing import Any, MutableMapping

SCOPE_PREFIX = "mp_page_"


def scoped_key(module: str, name: str) -> str:
    """Session/widget key for ``name`` inside ``module`` (always prefixed)."""
    return f"{SCOPE_PREFIX}{module}_{name}"


def set_scoped(state: MutableMapping, dataset_id: str, module: str, name: str, value: Any) -> None:
    state[scoped_key(module, name)] = {"dataset_id": dataset_id, "value": value}


def get_scoped(state: MutableMapping, dataset_id: str, module: str, name: str, default: Any = None) -> Any:
    """Return the stored value only if it belongs to ``dataset_id``; a value
    from another dataset is discarded and ``default`` returned."""
    key = scoped_key(module, name)
    entry = state.get(key)
    if not isinstance(entry, dict) or "dataset_id" not in entry:
        return default
    if entry["dataset_id"] != dataset_id:
        try:
            del state[key]
        except KeyError:
            pass
        return default
    return entry.get("value", default)


def clear_module(state: MutableMapping, module: str) -> int:
    """Remove every key belonging to ``module``; returns how many were removed."""
    prefix = scoped_key(module, "")
    doomed = [k for k in list(state.keys()) if isinstance(k, str) and k.startswith(prefix)]
    for k in doomed:
        del state[k]
    return len(doomed)
