"""Shared vocabulary for business-role detection (PART 02).

Plain data plus two tiny helpers, kept separate from ``role_engine`` so the
legacy detectors (``column_classifier`` and ``src.marketing.field_discovery``)
can reuse the same trade/promotion rule without importing the whole engine.

Nothing here names a specific dataset's columns: these are generic business
words, matched as whole tokens of a column name.
"""

from __future__ import annotations

import re

__all__ = ["tokens", "channel_label", "is_trade_or_promotion_spend_name", "CHANNEL_WORDS",
           "SPEND_WORDS", "TRADE_WORDS"]

# token -> display label of the advertising channel
CHANNEL_WORDS: dict[str, str] = {
    "tv": "TV", "television": "TV", "ctv": "Connected TV", "ott": "OTT", "cable": "TV",
    "radio": "Radio", "fm": "Radio", "podcast": "Podcast", "podcasts": "Podcast", "audio": "Audio",
    "print": "Print", "newspaper": "Print", "newspapers": "Print", "magazine": "Print", "magazines": "Print",
    "press": "Print",
    "search": "Search", "sem": "Search", "ppc": "Search", "google": "Google", "bing": "Bing", "adwords": "Search",
    "social": "Social", "facebook": "Facebook", "fb": "Facebook", "meta": "Meta", "instagram": "Instagram",
    "ig": "Instagram", "insta": "Instagram", "youtube": "YouTube", "yt": "YouTube", "tiktok": "TikTok",
    "linkedin": "LinkedIn", "twitter": "Twitter/X", "snapchat": "Snapchat", "snap": "Snapchat",
    "pinterest": "Pinterest", "whatsapp": "WhatsApp",
    "display": "Display", "banner": "Display", "programmatic": "Programmatic", "video": "Video",
    "ooh": "Outdoor", "outdoor": "Outdoor", "billboard": "Outdoor", "billboards": "Outdoor", "dooh": "Outdoor",
    "cinema": "Cinema", "email": "Email", "newsletter": "Email", "sms": "SMS", "affiliate": "Affiliate",
    "affiliates": "Affiliate", "influencer": "Influencer", "influencers": "Influencer",
    "digital": "Digital", "online": "Digital", "web": "Web", "website": "Website", "site": "Website",
    "mobile": "Mobile", "app": "App", "native": "Native", "retargeting": "Retargeting",
    "traditional": "Traditional", "offline": "Offline", "direct": "Direct mail", "mail": "Direct mail",
    "sponsorship": "Sponsorship", "events": "Events", "event": "Events",
}

# money words that make a column a spend/cost column
SPEND_WORDS = {"spend", "spends", "spending", "spent", "cost", "costs", "investment", "investments", "invest",
               "budget", "budgets", "expense", "expenses", "expenditure", "expenditures", "outlay", "adspend"}

# words that make a spend column trade / promotion expenditure rather than advertising
TRADE_WORDS = {"trade", "promo", "promos", "promotion", "promotions", "promotional", "tpr", "rebate", "rebates",
               "coupon", "coupons", "slotting", "merchandising", "deal", "deals", "markdown", "markdowns",
               "discount", "discounts", "trademarketing"}

_COMPOUNDS = {"youtube", "tiktok", "linkedin", "facebook", "instagram", "whatsapp", "pageviews", "snapchat",
              "adwords", "weekofyear", "dayofweek"}
_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def tokens(name: str) -> list[str]:
    """Lower-case word tokens of a column name: ``"TV_Impressions"`` -> ``["tv", "impressions"]``,
    ``"netSalesValue"`` -> ``["net", "sales", "value"]``."""
    s = _CAMEL.sub(r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    raw = [t for t in _SPLIT.split(s.lower()) if t]
    # re-join camel-case brand words split above ("YouTube" -> "you", "tube" -> "youtube")
    out: list[str] = []
    i = 0
    while i < len(raw):
        if i + 1 < len(raw) and (raw[i] + raw[i + 1]) in _COMPOUNDS:
            out.append(raw[i] + raw[i + 1])
            i += 2
        else:
            out.append(raw[i])
            i += 1
    return out


def channel_label(toks: list[str]) -> str | None:
    """First advertising-channel word in the tokens, as a display label."""
    for t in toks:
        if t in CHANNEL_WORDS:
            return CHANNEL_WORDS[t]
    return None


def is_trade_or_promotion_spend_name(name: str) -> bool:
    """True when a column name describes trade / promotion expenditure (e.g. trade spend,
    promotion budget, coupon cost). Such columns are business variables, not channel-wise
    advertising spend, unless a user explicitly confirms otherwise."""
    toks = set(tokens(name))
    joined = "".join(tokens(name))
    if toks & TRADE_WORDS:
        return True
    return "tradespend" in joined or "promospend" in joined
