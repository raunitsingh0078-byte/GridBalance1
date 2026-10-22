import copy
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from load_data import DEFAULT_REGION, TARGET, get_region_label, resolve_region
from preprocessing import FEATURE_COLUMNS, prepare_dataset


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

# Hourly rows kept for training, roughly the most recent 3.5 years
TRAIN_WINDOW = 30000

TEST_SIZE = 0.2

RANDOM_STATE = 42


def get_models() -> dict:
    """Candidate regressors, each wrapped in the same scaling pipeline."""

    models = {

        "Linear Regression":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]),

        "Ridge Regression":
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]),

        "Random Forest":
            Pipeline([
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80,
                        max_depth=16,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                ),
            ]),

        "Gradient Boosting":
            Pipeline([
                ("scaler", StandardScaler()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.1,
                        max_depth=None,
                        random_state=RANDOM_STATE,
                    )
                ),
            ]),
    }

    return models


# =========================================================
# METRICS
# =========================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    # Demand is always well above zero, so the guard only catches stray zeros
    non_zero = y_true != 0

    if non_zero.any():

        mape = np.mean(
            np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])
        ) * 100

    else:

        mape = float("nan")

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "mape": round(float(mape), 2),
    }


# =========================================================
# BASELINES
# =========================================================

def evaluate_baselines(y_true: np.ndarray, previous_hour: np.ndarray,
                       previous_week: np.ndarray) -> dict:
    """Simple reference points, so the ML scores can be judged."""

    return {

        "Naive (previous hour)":
            evaluate(y_true, previous_hour),

        "Seasonal Naive (previous week)":
            evaluate(y_true, previous_week),
    }


# =========================================================
# TRAINING
# =========================================================

# Benchmark results per region, so revisiting the page is instant
_BENCHMARKS = {}


def clear_benchmark_cache() -> None:
    """Drop cached benchmark results, for example between edits."""

    _BENCHMARKS.clear()


def train_models(region: str = DEFAULT_REGION,
                 train_window: int = TRAIN_WINDOW) -> dict:

    region = resolve_region(region)

    cache_key = (region, int(train_window))

    if cache_key in _BENCHMARKS:

        return copy.deepcopy(_BENCHMARKS[cache_key])

    prepared = prepare_dataset(region)

    if len(prepared) > train_window:

        prepared = prepared.tail(train_window).reset_index(drop=True)

    X = prepared[FEATURE_COLUMNS]

    y = prepared[TARGET]

    # Chronological split, never shuffled, so the future is never trained on
    split_index = int(len(prepared) * (1 - TEST_SIZE))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Baselines reuse the lag columns already present in the test window
    previous_hour = X_test["demand_lag_1"].to_numpy()
    previous_week = X_test["demand_lag_168"].to_numpy()

    results = {}

    timings = {}

    for name, model in get_models().items():

        started = time.perf_counter()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        elapsed = time.perf_counter() - started

        results[name] = evaluate(y_test.to_numpy(), predictions)

        timings[name] = round(elapsed, 2)

    baselines = evaluate_baselines(
        y_test.to_numpy(),
        previous_hour,
        previous_week
    )

    # Lowest RMSE wins
    best_model = min(results, key=lambda k: results[k]["rmse"])

    actual_mean = float(y_test.mean())

    return_value = {

        "region": region,
        "region_label": get_region_label(region),

        "target": TARGET,
        "target_unit": "MW",

        "results": results,

        "baselines": baselines,

        "timings": timings,

        "best_model": best_model,

        "best_rmse": results[best_model]["rmse"],
        "best_r2": results[best_model]["r2"],
        "best_mape": results[best_model]["mape"],

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "features": X.shape[1],

        "train_window": len(prepared),

        "test_size": TEST_SIZE,

        "average_demand": round(actual_mean, 2),

        "peak_demand": round(float(y.max()), 2),

        "minimum_demand": round(float(y.min()), 2),

        "feature_columns": FEATURE_COLUMNS,
    }

    _BENCHMARKS[cache_key] = return_value

    return copy.deepcopy(return_value)


# =========================================================
# MODEL CACHE FOR THE FORECAST PAGE
# =========================================================

_TRAINED = {}


def get_trained_model(region: str = DEFAULT_REGION):
    """Train once per region and reuse for the forecast page."""

    region = resolve_region(region)

    if region in _TRAINED:

        return _TRAINED[region]

    prepared = prepare_dataset(region)

    if len(prepared) > TRAIN_WINDOW:

        prepared = prepared.tail(TRAIN_WINDOW).reset_index(drop=True)

    model = get_models()["Gradient Boosting"]

    model.fit(prepared[FEATURE_COLUMNS], prepared[TARGET])

    _TRAINED[region] = (model, prepared)

    return _TRAINED[region]


if __name__ == "__main__":

    import json

    outcome = train_models()

    print(json.dumps(outcome, indent=2, default=str))
