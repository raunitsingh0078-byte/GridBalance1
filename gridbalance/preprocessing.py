from pathlib import Path

import numpy as np
import pandas as pd

from load_data import (
    DATA_DIR,
    DATE_COLUMN,
    DEFAULT_REGION,
    TARGET,
    get_region_label,
    get_value_column,
    load_data,
    resolve_region,
)


# =========================================================
# FEATURE CONFIGURATION
# =========================================================

LAGS = (1, 24, 168)

ROLLING_WINDOWS = (24, 168)

# Rows created for the most recent lag so lag features stay usable
LONGEST_LAG = max(LAGS)

FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "dayofyear",
    "month",
    "quarter",
    "day",
    "is_weekend",
    "is_peak_hour",
    "is_holiday_window",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
    "trend_index",
    "demand_lag_1",
    "demand_lag_24",
    "demand_lag_168",
    "demand_rolling_24",
    "demand_rolling_168",
]


# =========================================================
# CLEANING
# =========================================================

def clean_series(df: pd.DataFrame) -> tuple:
    """Turn a raw hourly archive into a gap-free, sorted, sane series."""

    report = {
        "timestamps_dropped": int(df[DATE_COLUMN].isna().sum()),
        "rows_reordered": 0,
        "duplicate_timestamps_removed": 0,
        "hours_filled": 0,
        "negative_values_fixed": 0,
        "outliers_clipped": 0,
        "lower_bound": 0.0,
        "upper_bound": 0.0,
    }

    cleaned = df.dropna(subset=[DATE_COLUMN]).copy()

    if not cleaned[DATE_COLUMN].is_monotonic_increasing:

        sorted_index = cleaned[DATE_COLUMN].sort_values().index

        report["rows_reordered"] = int(
            (sorted_index.to_numpy() != cleaned.index.to_numpy()).sum()
        )

        cleaned = cleaned.sort_values(DATE_COLUMN)

    duplicate_timestamps = int(cleaned[DATE_COLUMN].duplicated().sum())

    if duplicate_timestamps > 0:

        report["duplicate_timestamps_removed"] = duplicate_timestamps

        cleaned = cleaned.drop_duplicates(
            subset=[DATE_COLUMN],
            keep="first"
        )

    # Rebuild a complete hourly index so gaps become real rows
    full_index = pd.date_range(
        cleaned[DATE_COLUMN].min(),
        cleaned[DATE_COLUMN].max(),
        freq="h"
    )

    reindexed = cleaned.set_index(DATE_COLUMN)[TARGET].reindex(full_index)

    report["hours_filled"] = int(reindexed.isna().sum())

    reindexed = reindexed.interpolate(method="linear").ffill().bfill()

    cleaned = pd.DataFrame(
        {
            DATE_COLUMN: full_index,
            TARGET: reindexed.to_numpy(),
        }
    )

    # Demand can never be negative
    negative_mask = cleaned[TARGET] < 0

    if negative_mask.any():

        report["negative_values_fixed"] = int(negative_mask.sum())

        cleaned.loc[negative_mask, TARGET] = 0.0

    # Clip extreme spikes using the interquartile range
    q1 = float(cleaned[TARGET].quantile(0.25))
    q3 = float(cleaned[TARGET].quantile(0.75))

    iqr = q3 - q1

    lower_bound = max(0.0, q1 - 3.0 * iqr)
    upper_bound = q3 + 3.0 * iqr

    report["lower_bound"] = round(lower_bound, 2)
    report["upper_bound"] = round(upper_bound, 2)

    clipped_mask = (cleaned[TARGET] < lower_bound) | (
        cleaned[TARGET] > upper_bound
    )

    if clipped_mask.any():

        report["outliers_clipped"] = int(clipped_mask.sum())

        cleaned[TARGET] = cleaned[TARGET].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    cleaned[TARGET] = cleaned[TARGET].astype(float)

    return cleaned.reset_index(drop=True), report


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the calendar, cyclical and lag features used by the models."""

    features = df.copy()

    timestamps = features[DATE_COLUMN]

    hour = timestamps.dt.hour
    dayofweek = timestamps.dt.dayofweek
    dayofyear = timestamps.dt.dayofyear
    month = timestamps.dt.month

    features["hour"] = hour
    features["dayofweek"] = dayofweek
    features["dayofyear"] = dayofyear
    features["month"] = month
    features["quarter"] = timestamps.dt.quarter
    features["day"] = timestamps.dt.day

    features["is_weekend"] = (dayofweek >= 5).astype(int)

    # Typical weekday morning and evening ramp
    features["is_peak_hour"] = (
        ((hour >= 7) & (hour <= 9) & (dayofweek < 5))
        | ((hour >= 17) & (hour <= 20) & (dayofweek < 5))
    ).astype(int)

    features["is_holiday_window"] = (
        (month == 12) & (timestamps.dt.day >= 23)
    ).astype(int)

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    features["weekday_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    # Day of year drifts slowly, so scale it across a whole year
    features["year_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    features["year_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

    features["trend_index"] = np.arange(len(features), dtype=float)

    for lag in LAGS:

        features[f"demand_lag_{lag}"] = features[TARGET].shift(lag)

    for window in ROLLING_WINDOWS:

        # Shifted by one hour so the average never sees the current reading
        features[f"demand_rolling_{window}"] = (
            features[TARGET]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
        )

    features = features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return features


def prepare_dataset(region: str = DEFAULT_REGION) -> pd.DataFrame:
    """Full load, clean and feature-engineering path for one region."""

    region = resolve_region(region)

    cleaned, _ = clean_series(load_data(region))

    return build_features(cleaned)


# =========================================================
# PREPROCESSING SUMMARY
# =========================================================

def get_preprocessing_summary(region: str = DEFAULT_REGION) -> dict:

    region = resolve_region(region)

    raw = load_data(region)

    # Missing values before processing
    missing_before = int(raw.isnull().sum().sum())

    # Duplicate rows
    duplicates_before = int(raw.duplicated().sum())

    cleaned, report = clean_series(raw)

    # The datetime column is parsed to a real datetime type,
    # so no categorical columns remain after processing
    categorical_columns = []

    processed = build_features(cleaned)

    missing_after = int(cleaned.isnull().sum().sum())

    return {

        "region": region,
        "region_label": get_region_label(region),
        "source_file": f"{region}.csv",
        "value_column": get_value_column(region),

        "original_rows": len(raw),

        "original_columns": raw.shape[1],

        "duplicates_removed": duplicates_before,

        "duplicate_timestamps_removed": report["duplicate_timestamps_removed"],

        "missing_values_before": missing_before,

        "missing_values_after": missing_after,

        "hours_filled": report["hours_filled"],

        "rows_reordered": report["rows_reordered"],

        "timestamps_dropped": report["timestamps_dropped"],

        "negative_values_fixed": report["negative_values_fixed"],

        "outliers_clipped": report["outliers_clipped"],

        "lower_bound": report["lower_bound"],

        "upper_bound": report["upper_bound"],

        "processed_rows": len(cleaned),

        "processed_columns": len(cleaned.columns),

        "modelling_rows": len(processed),

        "modelling_columns": len(processed.columns),

        "numerical_columns": FEATURE_COLUMNS,

        "categorical_columns": categorical_columns,

        "feature_count": len(FEATURE_COLUMNS),

        "target_column": TARGET,

        "feature_preview": (
            processed[FEATURE_COLUMNS + [TARGET]]
            .head(10)
            .round(3)
            .to_dict(orient="records")
        ),

        "feature_table": [
            {
                "feature": column,
                "group": feature_group(column),
            }
            for column in FEATURE_COLUMNS + [TARGET]
        ],
    }


def feature_group(column: str) -> str:
    """Label each column so the template can group the feature list."""

    if column == TARGET:
        return "Target"

    if column.startswith("demand_lag_"):
        return "Lag Features"

    if column.startswith("demand_rolling_"):
        return "Rolling Features"

    if column.endswith("_sin") or column.endswith("_cos"):
        return "Cyclical Features"

    if column == "trend_index":
        return "Trend Feature"

    if column.startswith("is_"):
        return "Binary Flags"

    return "Calendar Features"


if __name__ == "__main__":

    import json

    summary = get_preprocessing_summary()

    print(json.dumps(summary, indent=2, default=str))
