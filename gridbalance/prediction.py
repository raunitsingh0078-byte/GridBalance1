import numpy as np
import pandas as pd

from load_data import (
    DATE_COLUMN,
    DEFAULT_REGION,
    TARGET,
    get_region_label,
    list_regions,
    load_data,
    resolve_region,
)
from model_training import get_trained_model, get_models
from preprocessing import (
    FEATURE_COLUMNS,
    LAGS,
    ROLLING_WINDOWS,
    clean_series,
)


# =========================================================
# FORECAST CONFIGURATION
# =========================================================

HORIZON_OPTIONS = [6, 12, 24, 48, 168]

DEFAULT_HORIZON = 24

# Used when the user leaves the model on automatic
AUTO_MODEL = "Gradient Boosting"

# Actual readings kept in memory so the lag features stay realistic
HISTORY_TAIL = 500


# =========================================================
# FEATURE BUILDING FOR FUTURE STEPS
# =========================================================

def build_future_features(future_times: pd.DatetimeIndex,
                          combined: list,
                          start_position: int,
                          trend_start: int) -> pd.DataFrame:
    """Recreate the model features for hours that have not happened yet.

    combined holds the recent actual readings followed by the forecasts
    already produced, so lag and rolling features stay consistent with
    build_features in preprocessing.py.
    """

    values = np.asarray(combined, dtype=float)

    count = len(future_times)

    hour = np.asarray(future_times.hour)
    dayofweek = np.asarray(future_times.dayofweek)
    dayofyear = np.asarray(future_times.dayofyear)
    month = np.asarray(future_times.month)

    features = pd.DataFrame(
        {
            "hour": hour,
            "dayofweek": dayofweek,
            "dayofyear": dayofyear,
            "month": month,
            "quarter": np.asarray(future_times.quarter),
            "day": np.asarray(future_times.day),
            "is_weekend": (dayofweek >= 5).astype(int),
            "is_peak_hour": (
                ((hour >= 7) & (hour <= 9) & (dayofweek < 5))
                | ((hour <= 20) & (hour >= 17) & (dayofweek < 5))
            ).astype(int),
            "is_holiday_window": (
                (month == 12) & (np.asarray(future_times.day) >= 23)
            ).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * dayofweek / 7),
            "weekday_cos": np.cos(2 * np.pi * dayofweek / 7),
            "year_sin": np.sin(2 * np.pi * dayofyear / 365.25),
            "year_cos": np.cos(2 * np.pi * dayofyear / 365.25),
            "trend_index": trend_start + np.arange(1, count + 1),
        }
    )

    for step in range(count):

        position = start_position + step

        for lag in LAGS:

            features.at[step, f"demand_lag_{lag}"] = values[position - lag]

        for window in ROLLING_WINDOWS:

            window_slice = values[position - window:position]

            features.at[step, f"demand_rolling_{window}"] = np.mean(
                window_slice
            )

    return features


# =========================================================
# FORECAST
# =========================================================

def forecast(region: str, horizon: int = DEFAULT_HORIZON,
             model_name: str = AUTO_MODEL) -> list:

    region = resolve_region(region)

    horizon = max(1, min(int(horizon), 168))

    cleaned, _ = clean_series(load_data(region))

    model, prepared = get_trained_model(region)

    if model_name and model_name != "Auto":

        # Fit the requested model on the same window the automatic one used
        model = get_models()[model_name]

        model.fit(prepared[FEATURE_COLUMNS], prepared[TARGET])

    # Recent actual readings, then forecasts are appended to this list
    combined = cleaned[TARGET].to_numpy(dtype=float)[-HISTORY_TAIL:].tolist()

    start_position = len(combined)

    trend_start = int(prepared["trend_index"].iloc[-1])

    future_times = pd.date_range(
        cleaned[DATE_COLUMN].iloc[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h"
    )

    # Step forward one hour at a time so each forecast can use the last one
    predictions = []

    for step in range(horizon):

        step_features = build_future_features(
            future_times[step:step + 1],
            combined,
            start_position + step,
            trend_start
        )

        value = float(model.predict(step_features)[0])

        value = max(0.0, value)

        combined.append(value)

        predictions.append(value)

    # Residual scale of the fitted model, used for the confidence band
    residuals = (
        prepared[TARGET].to_numpy()
        - model.predict(prepared[FEATURE_COLUMNS])
    )

    sigma = float(np.std(residuals[-5000:]))

    last_actual = float(cleaned[TARGET].iloc[-1])

    weekdays = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    rows = []

    previous = last_actual

    for step, value in enumerate(predictions):

        timestamp = future_times[step]

        rows.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                "hour": timestamp.strftime("%H:00"),
                "weekday": weekdays[timestamp.dayofweek],
                "predicted_demand": round(value, 2),
                "lower_bound": round(max(0.0, value - 2 * sigma), 2),
                "upper_bound": round(value + 2 * sigma, 2),
                "change": round(value - previous, 2),
                "change_percent": round(
                    100 * (value - previous) / previous, 2
                ) if previous else 0.0,
            }
        )

        previous = value

    return rows


# =========================================================
# PREDICTION ENTRY POINT
# =========================================================

def predict_demand(form) -> dict:

    region = form.get("region", DEFAULT_REGION)

    horizon = form.get("horizon", DEFAULT_HORIZON)

    model_name = form.get("model", AUTO_MODEL) or AUTO_MODEL

    try:

        horizon = int(horizon)

    except (TypeError, ValueError):

        horizon = DEFAULT_HORIZON

    rows = forecast(region, horizon, model_name)

    predicted = [row["predicted_demand"] for row in rows]

    cleaned, _ = clean_series(load_data(resolve_region(region)))

    recent_average = float(cleaned[TARGET].tail(168).mean())

    forecast_average = float(np.mean(predicted))

    return {

        "region": region,
        "region_label": get_region_label(region),

        "model": model_name if model_name != "Auto" else AUTO_MODEL,

        "horizon": horizon,

        "forecast_start": rows[0]["timestamp"],
        "forecast_end": rows[-1]["timestamp"],

        "rows": rows,

        "peak_demand": round(max(predicted), 2),
        "peak_hour": max(rows, key=lambda r: r["predicted_demand"])["hour"],

        "minimum_demand": round(min(predicted), 2),
        "minimum_hour": min(rows, key=lambda r: r["predicted_demand"])["hour"],

        "average_demand": round(forecast_average, 2),

        "total_energy_mwh": round(float(np.sum(predicted)), 2),

        "recent_average": round(recent_average, 2),

        "difference_percent": round(
            100 * (forecast_average - recent_average) / recent_average, 2
        ) if recent_average else 0.0,
    }


def get_forecast_form_options() -> dict:
    """Choices the template needs to build the forecast form."""

    return {
        "regions": list_regions(),
        "horizons": HORIZON_OPTIONS,
        "models": ["Auto"] + list(get_models().keys()),
        "default_region": DEFAULT_REGION,
        "default_horizon": DEFAULT_HORIZON,
    }


if __name__ == "__main__":

    import json

    print(json.dumps(predict_demand(
        {"region": "AEP_hourly", "horizon": "24", "model": "Auto"}
    ), indent=2, default=str))
