"""Alias vocabularies for semantic column-role detection.

Every set holds LOWERCASE, alphanumeric-normalized tokens (the classifier
normalizes incoming column names the same way before matching: lowercase,
strip surrounding whitespace, replace separators with a single space).
Nothing here references any specific dataset's exact schema — these are
generic, reusable vocabularies.
"""

DATE_ALIASES = {
    "date", "week", "day", "month", "period", "timestamp", "time",
    "order date", "transaction date", "week starting", "week ending",
    "report date", "activity date", "interaction date", "created at",
    "created date", "event date", "date recorded",
}

REVENUE_ALIASES = {
    "revenue", "total revenue", "sales revenue", "order value",
    "amount generated", "sales amount", "transaction amount",
    "total amount", "revenue usd", "net revenue", "gross revenue",
    "income", "turnover", "total sales value", "order amount",
    "sale amount", "amount", "value generated", "revenue generated",
    # financial "sales" terms that mean money, not units
    "net sales", "gross sales", "sales value", "net sales value",
}

SALES_ALIASES = {
    "sales", "units sold", "sales units", "sales volume", "orders sold",
    "quantity sold", "sales count", "number sold",
}

COST_ALIASES = {
    "cost", "total cost", "cost of goods sold", "cogs", "expense",
    "expenses", "operating cost", "total expense",
}

# Per-channel spend vocabularies — union of all of these forms SPEND_ALIASES.
CHANNEL_SPEND_ALIASES = {
    "google": {"google ads", "google_ads", "google spend", "search ads", "sem spend", "adwords",
               "google advertising", "google ad spend", "google ad cost"},
    "facebook": {"facebook ads", "facebook_ads", "facebook spend", "fb ads", "fb spend",
                 "facebook advertising", "meta ads", "meta spend", "meta advertising", "meta ad spend"},
    "instagram": {"instagram ads", "instagram_ads", "instagram spend", "ig spend"},
    "youtube": {"youtube ads", "youtube_ads", "youtube spend", "yt spend"},
    "tv": {"tv spend", "tv ads", "television spend", "television ads"},
    "radio": {"radio spend", "radio ads"},
    "print": {"print spend", "print ads", "newspaper spend", "magazine spend"},
    "search": {"search spend", "search ads", "ppc spend", "sem spend", "search ad cost",
               "search advertising", "ppc cost"},
    "social": {"social spend", "social media spend", "social ads"},
    "display": {"display spend", "display ads", "banner spend"},
    "email": {"email spend", "email marketing spend", "email ads"},
    "influencer": {"influencer spend", "influencer marketing spend"},
    "tiktok": {"tiktok ads", "tiktok_ads", "tiktok spend"},
    "linkedin": {"linkedin ads", "linkedin_ads", "linkedin spend"},
    "twitter": {"twitter ads", "twitter_ads", "twitter spend", "x ads", "x spend"},
}

# "<channel> cost" / "<channel> budget" are spend too (e.g. facebook_cost, tv_budget).
# Without these, "facebook cost" matched the generic business-cost (COGS) role.
for _brand, _extra in {"google": ("google",), "facebook": ("facebook", "fb", "meta"),
                       "instagram": ("instagram",), "youtube": ("youtube",), "tv": ("tv", "television"),
                       "radio": ("radio",), "print": ("print",), "search": ("search", "ppc", "sem"),
                       "social": ("social", "social media"), "display": ("display",), "email": ("email",),
                       "influencer": ("influencer",), "tiktok": ("tiktok",), "linkedin": ("linkedin",),
                       "twitter": ("twitter",)}.items():
    for _w in _extra:
        CHANNEL_SPEND_ALIASES[_brand] |= {f"{_w} cost", f"{_w} ad cost", f"{_w} ads cost", f"{_w} budget"}

SPEND_ALIASES = set()
for _vals in CHANNEL_SPEND_ALIASES.values():
    SPEND_ALIASES |= _vals
# generic spend words (not channel-specific but still "this looks like spend")
SPEND_GENERIC_ALIASES = {"spend", "ad spend", "marketing spend", "media spend", "budget spent", "campaign spend",
                         "marketing investment", "media investment", "advertising spend", "advertising cost",
                         "ad cost", "marketing cost"}

IMPRESSIONS_ALIASES = {"impressions", "ad impressions", "views", "reach", "impression count"}
CLICKS_ALIASES = {"clicks", "ad clicks", "click count", "link clicks"}
CONVERSIONS_ALIASES = {"conversions", "conversion count", "leads", "sign ups", "signups"}
ORDERS_ALIASES = {"orders", "order count", "num orders", "number of orders", "total orders"}
PROFIT_ALIASES = {"profit", "net profit", "gross profit", "total profit", "profit amount"}
MARGIN_ALIASES = {"margin", "profit margin", "gross margin", "margin percent", "margin pct"}
DISCOUNT_ALIASES = {"discount", "discount amount", "discount pct", "discount percent", "coupon value"}
QUANTITY_ALIASES = {"quantity", "qty", "units", "unit count", "item count", "quantity sold"}
RATING_ALIASES = {"rating", "review rating", "star rating", "score", "csat", "nps"}
DURATION_ALIASES = {"duration", "duration minutes", "duration seconds", "session duration", "time spent", "call duration"}
ENGAGEMENT_ALIASES = {"engagement", "engagement rate", "engagement score", "interaction count"}
CAMPAIGN_ALIASES = {"campaign", "campaign name", "campaign id"}
CHANNEL_ALIASES = {"channel", "marketing channel", "acquisition channel", "traffic source", "source", "medium"}
REGION_ALIASES = {"region", "state", "country", "city", "location", "territory", "market"}
PRODUCT_ALIASES = {"product", "product name", "product category", "category", "sku", "item"}
IDENTIFIER_ALIASES = {
    "id", "user id", "customer id", "client id", "account id", "order id",
    "transaction id", "record id", "employee id", "row id", "uuid",
}

# key -> alias set, used by the column classifier to iterate all roles.
ALIAS_REGISTRY = {
    "date": DATE_ALIASES,
    "revenue": REVENUE_ALIASES,
    "sales": SALES_ALIASES,
    "cost": COST_ALIASES,
    "impressions": IMPRESSIONS_ALIASES,
    "clicks": CLICKS_ALIASES,
    "conversions": CONVERSIONS_ALIASES,
    "orders": ORDERS_ALIASES,
    "profit": PROFIT_ALIASES,
    "margin": MARGIN_ALIASES,
    "discount": DISCOUNT_ALIASES,
    "quantity": QUANTITY_ALIASES,
    "rating": RATING_ALIASES,
    "duration": DURATION_ALIASES,
    "engagement": ENGAGEMENT_ALIASES,
    "campaign": CAMPAIGN_ALIASES,
    "channel_dim": CHANNEL_ALIASES,
    "region": REGION_ALIASES,
    "product": PRODUCT_ALIASES,
    "identifier": IDENTIFIER_ALIASES,
}
