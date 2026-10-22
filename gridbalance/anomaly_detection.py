from statistics import NormalDist

import numpy as np
import pandas as pd

from load_data import (
    DATE_COLUMN,
    DEFAULT_REGION,
    TARGET,
    get_region_label,
    list_regions,
    resolve_region,
)
from preprocessing import clean_series, load_data


# =========================================================
# DETECTION CONFIGURATION
# =========================================================

# Days of history used to learn the normal shape of the load curve
WINDOW_DAYS = 365

DEFAULT_SENSITIVITY = 0.01

# Keeps the flag rate sensible even for very loose sensitivity values
MINIMUM_THRESHOLD = 1.5

# A meter repeating one value for this many hours is treated as stuck
STUCK_HOURS = 6

SENSITIVITY_OPTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]

MAX_ROWS = 100


def get_reference_threshold(sensitivity: float) -> float:
    """Normal theory cut off for the requested flag rate.

    Reported for comparison only. Seasonal load data is far too heavy
    tailed for this value to be used directly.
    """

    sensitivity = min(max(float(sensitivity), 0.0005), 0.25)

    threshold = NormalDist().inv_cdf(1 - sensitivity / 2)

    return round(max(threshold, MINIMUM_THRESHOLD), 2)


def get_threshold(z_scores: pd.Series, sensitivity: float) -> float:
    """Pick the robust z-score cut off that matches the requested rate.

    Taking the empirical quantile means sensitivity behaves exactly as
    advertised, so the control keeps working at every setting.
    """

    sensitivity = min(max(float(sensitivity), 0.0005), 0.25)

    observed = z_scores.abs().dropna()

    if len(observed) == 0:

        return MINIMUM_THRESHOLD

    threshold = float(observed.quantile(1 - sensitivity))

    return round(max(threshold, MINIMUM_THRESHOLD), 2)


# =========================================================
# DETECTION
# =========================================================

def detect_anomalies(region: str = DEFAULT_REGION,
                     sensitivity: float = DEFAULT_SENSITIVITY,
                     limit: int = MAX_ROWS) -> dict:

    region = resolve_region(region)

    cleaned, _ = clean_series(load_data(region))

    window_hours = min(len(cleaned), WINDOW_DAYS * 24)

    work = cleaned.tail(window_hours).copy()

    timestamps = work[DATE_COLUMN]

    # Each weekday-hour pair gets its own expected level
    work["slot"] = timestamps.dt.dayofweek * 24 + timestamps.dt.hour

    grouped = work.groupby("slot")

    work["expected"] = grouped[TARGET].transform("median")

    work["absolute_deviation"] = (work[TARGET] - work["expected"]).abs()

    # Median absolute deviation, scaled to match a normal distribution
    work["scale"] = 1.4826 * grouped["absolute_deviation"].transform(
        "median"
    )

    # Fall back to the standard deviation when a slot is perfectly flat
    fallback_scale = grouped[TARGET].transform("std")

    work["scale"] = np.where(
        work["scale"] > 0,
        work["scale"],
        fallback_scale
    )

    work["z_score"] = (
        (work[TARGET] - work["expected"])
        / work["scale"].replace(0, np.nan)
    )

    work["deviation"] = work[TARGET] - work["expected"]

    # Chosen after scoring so the cut off matches the requested flag rate
    threshold = get_threshold(work["z_score"], sensitivity)

    # Meter faults: impossible readings and a frozen sensor
    work["impossible"] = work[TARGET] <= 0

    block = (work[TARGET].shift() != work[TARGET]).cumsum()

    run_length = work.groupby(block)[TARGET].transform("size")

    work["stuck"] = run_length >= STUCK_HOURS

    work["spike"] = work["z_score"].abs() > threshold

    work["flagged"] = work["impossible"] | work["stuck"] | work["spike"]

    # Highest severity first, so the most serious reason is always kept
    def reason(row):

        if row["impossible"]:
            return "Impossible reading (zero or negative demand)"

        if row["stuck"]:
            return f"Meter stuck on one value for {STUCK_HOURS}+ hours"

        if row["z_score"] is not None and np.isnan(row["z_score"]):
            return "Unstable baseline"

        if row["z_score"] > 0:
            return "Demand spike against normal pattern"

        return "Demand dip against normal pattern"

    flagged = work[work["flagged"]].copy()

    if len(flagged) > 0:

        flagged["reason"] = flagged.apply(reason, axis=1)

        # Scaled against the worst reading in the window, so the most
        # extreme anomaly always scores 1.0
        worst_deviation = float(work["z_score"].abs().max())

        scale = worst_deviation if worst_deviation > 0 else 1.0

        flagged["severity"] = (
            flagged["z_score"].abs() / scale
        ).clip(0, 1).fillna(1.0)

        flagged.loc[flagged["impossible"], "severity"] = 1.0

        flagged.loc[flagged["stuck"], "severity"] = flagged.loc[
            flagged["stuck"], "severity"
        ].clip(lower=0.7)

        flagged = flagged.sort_values(
            by=["severity", "z_score"],
            ascending=[False, False],
            key=lambda column: (
                column.abs() if column.name == "z_score" else column
            )
        )

    rows = []

    for _, row in flagged.head(limit).iterrows():

        z_score = row["z_score"]

        rows.append(
            {
                "timestamp": row[DATE_COLUMN].strftime("%Y-%m-%d %H:%M"),
                "weekday": row[DATE_COLUMN].strftime("%A"),
                "hour": row[DATE_COLUMN].strftime("%H:00"),
                "demand": round(float(row[TARGET]), 2),
                "expected": round(float(row["expected"]), 2),
                "deviation": round(float(row["deviation"]), 2),
                "z_score": round(float(z_score), 2)
                if pd.notna(z_score)
                else None,
                "severity": round(float(row["severity"]), 3),
                "reason": row["reason"],
            }
        )

    reason_counts = {}

    if len(flagged) > 0:

        counts = flagged["reason"].value_counts()

        reason_counts = {reason: int(count) for reason, count in counts.items()}

    total_points = len(work)

    anomaly_count = int(work["flagged"].sum())

    pattern_count = int(work["spike"].sum())

    return {

        "region": region,
        "region_label": get_region_label(region),

        "sensitivity": sensitivity,
        "threshold": threshold,
        "reference_threshold": get_reference_threshold(sensitivity),

        "window_days": round(window_hours / 24, 1),

        "total_points": total_points,

        "anomaly_count": anomaly_count,

        "anomaly_rate": round(100 * anomaly_count / total_points, 3)
        if total_points
        else 0.0,

        "pattern_rate": round(100 * pattern_count / total_points, 3)
        if total_points
        else 0.0,

        "impossible_readings": int(work["impossible"].sum()),

        "stuck_readings": int(work["stuck"].sum()),

        "pattern_deviations": pattern_count,

        "reason_counts": reason_counts,

        "rows": rows,

        "rows_shown": len(rows),

        "peak_expected": round(float(work["expected"].max()), 2),
        "mean_demand": round(float(work[TARGET].mean()), 2),
    }


def get_anomaly_form_options() -> dict:
    """Choices the template needs to build the anomaly form."""

    return {
        "regions": list_regions(),
        "sensitivities": SENSITIVITY_OPTIONS,
        "default_region": DEFAULT_REGION,
        "default_sensitivity": DEFAULT_SENSITIVITY,
    }


if __name__ == "__main__":

    import json

    print(json.dumps(
        detect_anomalies("AEP_hourly", 0.01),
        indent=2,
        default=str
    ))
