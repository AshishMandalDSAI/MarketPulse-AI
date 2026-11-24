"""Currency detection for uploaded datasets (upgrade PART 01).

Rule: report a currency only when the evidence is unambiguous. Evidence comes from
column names (``Revenue (INR)``, ``amount_usd``, ``sales_eur``) and from the raw
cell text of monetary-looking columns (``₹1,200``, ``$12.50``, ``EUR 300``).

* A symbol shared by several currencies (``$``, ``¥``, ``Rs``, ``kr``) identifies the
  symbol but not the ISO code: the result keeps the symbol and leaves ``code``
  empty rather than guessing (``$`` is not assumed to be US dollars).
* Different currencies in different columns, or in one column, are a conflict:
  the result is "no currency", with the conflict listed.
* No evidence means no currency. INR is never assumed for an upload; the built-in
  demo dataset is configured as INR separately (``DEMO_CURRENCY``).

The user can always override the result on the Upload Data page; the choice is
stored with the dataset (``currency.json`` in its processed folder).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# ISO code -> (symbol used for display, name)
CURRENCIES: dict = {
    "INR": ("₹", "Indian rupee"), "USD": ("$", "US dollar"), "EUR": ("€", "Euro"),
    "GBP": ("£", "Pound sterling"), "JPY": ("¥", "Japanese yen"), "CNY": ("¥", "Chinese yuan"),
    "AUD": ("A$", "Australian dollar"), "CAD": ("C$", "Canadian dollar"), "SGD": ("S$", "Singapore dollar"),
    "HKD": ("HK$", "Hong Kong dollar"), "NZD": ("NZ$", "New Zealand dollar"), "CHF": ("CHF ", "Swiss franc"),
    "AED": ("AED ", "UAE dirham"), "SAR": ("SAR ", "Saudi riyal"), "ZAR": ("R ", "South African rand"),
    "BRL": ("R$", "Brazilian real"), "MXN": ("MX$", "Mexican peso"), "KRW": ("₩", "South Korean won"),
    "RUB": ("₽", "Russian ruble"), "TRY": ("₺", "Turkish lira"), "NGN": ("₦", "Nigerian naira"),
    "PHP": ("₱", "Philippine peso"), "IDR": ("Rp ", "Indonesian rupiah"), "MYR": ("RM ", "Malaysian ringgit"),
    "THB": ("฿", "Thai baht"), "BDT": ("৳", "Bangladeshi taka"), "PKR": ("PKR ", "Pakistani rupee"),
    "LKR": ("LKR ", "Sri Lankan rupee"), "NPR": ("NPR ", "Nepalese rupee"), "SEK": ("SEK ", "Swedish krona"),
    "NOK": ("NOK ", "Norwegian krone"), "DKK": ("DKK ", "Danish krone"), "PLN": ("zł ", "Polish zloty"),
    "ILS": ("₪", "Israeli shekel"), "VND": ("₫", "Vietnamese dong"), "EGP": ("EGP ", "Egyptian pound"),
    "KES": ("KES ", "Kenyan shilling"),
}

# Symbols that identify exactly one currency.
UNIQUE_SYMBOLS = {"₹": "INR", "€": "EUR", "£": "GBP", "₩": "KRW", "₽": "RUB", "₺": "TRY", "₦": "NGN",
                  "₱": "PHP", "฿": "THB", "৳": "BDT", "₪": "ILS", "₫": "VND",
                  "A$": "AUD", "C$": "CAD", "S$": "SGD", "HK$": "HKD", "NZ$": "NZD", "R$": "BRL",
                  "US$": "USD", "MX$": "MXN"}
# Symbols shared by several currencies: the symbol is known, the code is not.
AMBIGUOUS_SYMBOLS = {"$": "dollar", "¥": "yen / yuan", "Rs": "rupee", "kr": "krona / krone"}

# Name tokens (lower case, whole word) -> ISO code or ambiguous symbol.
_NAME_TOKENS = {code.lower(): code for code in CURRENCIES}
_NAME_TOKENS.update({"rupee": "Rs", "rupees": "Rs", "rs": "Rs", "inr": "INR", "dollar": "$", "dollars": "$",
                     "usd": "USD", "euro": "EUR", "euros": "EUR", "yen": "¥", "yuan": "CNY", "rmb": "CNY",
                     "sterling": "GBP"})
# Tokens that are also ordinary words or units in column names are only accepted
# when they stand inside brackets, e.g. "Price (try)" is not Turkish lira.
_BRACKETED_ONLY = {"try", "all", "top", "cup", "mad", "bob", "php", "sar", "aed", "nok", "egp", "ils",
                   "cad", "pln", "kes", "chf", "sek", "dkk", "npr", "idr", "rs"}

_VALUE_SYMBOL_RE = re.compile(r"(US\$|A\$|C\$|S\$|HK\$|NZ\$|R\$|MX\$|[₹€£₩₽₺₦₱฿৳₪₫$¥]|\bRs\.?(?=\s?\d))")
_VALUE_CODE_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(CURRENCIES) + r")(?![A-Za-z])")
_NUMBER_RE = re.compile(r"\d")

DEMO_CURRENCY = {"code": "INR", "symbol": "₹", "name": "Indian rupee", "source": "demo configuration"}


@dataclass
class CurrencyResult:
    code: Optional[str] = None          # ISO 4217 code when identified exactly
    symbol: Optional[str] = None        # display symbol when known (also for ambiguous symbols)
    name: Optional[str] = None
    source: str = "none"                # column names | cell values | names and values | user | none
    evidence: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    note: str = ""

    @property
    def identified(self) -> bool:
        return bool(self.symbol)

    def to_dict(self) -> dict:
        return asdict(self)


def _token_hits(name: str) -> list:
    lower = name.lower()
    hits = []
    for m in re.finditer(r"[a-z]+", lower.replace("_", " ")):
        tok = m.group(0)
        if tok in _NAME_TOKENS:
            start, end = m.start(), m.end()
            bracketed = (start > 0 and lower[start - 1] in "([") or (end < len(lower) and lower[end:end + 1] in ")]")
            if tok in _BRACKETED_ONLY and not bracketed:
                continue
            hits.append(_NAME_TOKENS[tok])
    for sym, code in UNIQUE_SYMBOLS.items():
        if sym in name and len(sym) == 1:
            hits.append(code)
    return hits


def _value_hits(series: pd.Series, sample: int = 500) -> list:
    if pd.api.types.is_numeric_dtype(series):
        return []
    text = series.dropna().astype(str)
    text = text[text.str.contains(_NUMBER_RE)].head(sample)
    hits = []
    for v in text:
        for m in _VALUE_SYMBOL_RE.finditer(v):
            s = m.group(0).rstrip(".")
            hits.append(UNIQUE_SYMBOLS.get(s, s))
        for m in _VALUE_CODE_RE.finditer(v):
            hits.append(m.group(0))
    return hits


def _resolve(token: str) -> tuple:
    """token (ISO code or ambiguous symbol) -> (code, symbol, name)."""
    if token in CURRENCIES:
        sym, name = CURRENCIES[token]
        return token, sym.strip() if sym.endswith(" ") else sym, name
    if token in AMBIGUOUS_SYMBOLS:
        return None, token, AMBIGUOUS_SYMBOLS[token] + " (country not identified)"
    return None, None, None


def detect_currency(df: pd.DataFrame, columns: Optional[Iterable] = None) -> CurrencyResult:
    """Detect the dataset's currency from column names and raw cell text.

    ``columns`` limits the search (e.g. to columns mapped as revenue, cost or spend);
    by default all columns are inspected. Run this on the *raw* upload: cleaning
    turns ``"₹1,200"`` into ``1200`` and removes the value evidence.
    """
    res = CurrencyResult()
    if df is None or df.empty:
        res.note = "No data to inspect."
        return res
    cols = [c for c in (columns if columns is not None else df.columns) if c in df.columns]
    per_col: dict = {}
    names_used = values_used = False
    for c in cols:
        name_hits = _token_hits(str(c))
        value_hits = _value_hits(df[c])
        tokens = set(name_hits) | set(value_hits)
        if not tokens:
            continue
        names_used |= bool(name_hits)
        values_used |= bool(value_hits)
        per_col[str(c)] = tokens
        res.evidence.append(f"{c}: " + ", ".join(sorted(tokens)) + (" (name)" if name_hits else "")
                            + (" (values)" if value_hits else ""))
    if not per_col:
        res.note = "No currency symbol or code was found in column names or values."
        return res

    # A column may hold e.g. {"$", "USD"}: an exact code refines an ambiguous symbol of the same family.
    def normalise(tokens: set) -> set:
        exact = {t for t in tokens if t in CURRENCIES}
        ambiguous = tokens - exact
        if exact:
            compatible = {a for a in ambiguous if all(_resolve(e)[1] and _resolve(e)[1].endswith(a) or
                                                      (a == "Rs" and e in ("INR", "PKR", "LKR", "NPR")) or
                                                      (a == "kr" and e in ("SEK", "NOK", "DKK"))
                                                      for e in exact)}
            return exact | (ambiguous - compatible)
        return ambiguous

    found = set()
    for c, toks in per_col.items():
        norm = normalise(toks)
        if len(norm) > 1:
            res.conflicts.append(f"'{c}' mixes {', '.join(sorted(norm))}")
        found |= norm
    found = normalise(found)
    if len(found) != 1 or res.conflicts:
        if len(found) > 1:
            res.conflicts.append("different currencies across columns: " + ", ".join(sorted(found)))
        res.note = "Conflicting currency evidence, so no currency is shown. Choose one on the Upload Data page if " \
                   "all monetary columns share a currency."
        return res
    token = next(iter(found))
    res.code, res.symbol, res.name = _resolve(token)
    res.source = "names and values" if names_used and values_used else ("column names" if names_used else "cell values")
    if res.code is None:
        res.note = f"The symbol '{res.symbol}' is used by several currencies; the exact currency is not identified."
    return res


# ---------------------------------------------------------------------------
# Persistence (per dataset)
# ---------------------------------------------------------------------------

def save_currency(processed_dir: Path, info: dict) -> Path:
    path = Path(processed_dir) / "currency.json"
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_currency(processed_dir: Path) -> Optional[dict]:
    path = Path(processed_dir) / "currency.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def choice_to_info(choice: str, detected: Optional[CurrencyResult], monetary_columns: Iterable = ()) -> dict:
    """Upload-page choice -> stored record. ``choice`` is ``"auto"``, ``"none"`` or an ISO code."""
    mon = [str(c) for c in monetary_columns]
    if choice == "none":
        return {"code": None, "symbol": None, "name": None, "source": "user: none / not monetary",
                "monetary_columns": mon}
    if choice == "auto":
        d = detected or CurrencyResult()
        return {"code": d.code, "symbol": d.symbol, "name": d.name,
                "source": f"auto-detected ({d.source})" if d.identified else "auto-detected: none found",
                "evidence": d.evidence, "conflicts": d.conflicts, "monetary_columns": mon}
    if choice not in CURRENCIES:
        raise ValueError(f"Unknown currency code {choice!r}")
    sym, name = CURRENCIES[choice]
    return {"code": choice, "symbol": sym.strip() if sym.endswith(" ") else sym, "name": name,
            "source": "user selection", "monetary_columns": mon}
