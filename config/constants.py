"""Reasonable-default thresholds used across the dataset-first platform."""

# Minimum number of spend/channel-like columns required for MMM to be viable.
MIN_CHANNELS_FOR_MMM = 2

# Minimum number of time points required to attempt a forecast.
MIN_WEEKS_FOR_FORECAST = 12

# Confidence thresholds for column-role guesses.
CONFIDENCE_HIGH = 0.80
CONFIDENCE_MEDIUM = 0.55
# anything below CONFIDENCE_MEDIUM is LOW.

# A column whose ratio of unique values to row count exceeds this is treated
# as an identifier (e.g. user_id, order_id) rather than a measure/dimension.
IDENTIFIER_CARDINALITY_RATIO = 0.9

# A categorical column with more distinct values than this is too high-
# cardinality to be treated as a useful "dimension" for grouping/breakdowns.
MAX_DIMENSION_CARDINALITY = 500
