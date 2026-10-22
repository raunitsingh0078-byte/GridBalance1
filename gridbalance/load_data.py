import os
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# The hourly CSV archives live directly inside the project folder
DATA_DIR = BASE_DIR


# =========================================================
# REGION CONFIGURATION
# =========================================================

DEFAULT_REGION = "PJM_Load_hourly"

TARGET = "Demand_MW"

DATE_COLUMN = "Datetime"

REGION_LABELS = {
    "PJM_Load_hourly": "PJM Interconnection Total Load",
    "AEP_hourly": "AEP",
    "COMED_hourly": "ComEd",
    "DAYTON_hourly": "Dayton",
    "DEOK_hourly": "DEOK",
    "DOM_hourly": "DOM",
    "DUQ_hourly": "DUQ",
    "EKPC_hourly": "EKPC",
    "FE_hourly": "FE",
    "NI_hourly": "NI",
    "PJME_hourly": "PJME",
    "PJMW_hourly": "PJMW",
}


# =========================================================
# COMBINED ARCHIVE
# =========================================================
# One wide file holding every regional series. Each column is only
# filled in over that region's own date range, so it is a cross check
# on the per region archives rather than a thirteenth region.

COMBINED_PARQUET = "est_hourly.parquet"

COMBINED_CSV_FALLBACK = "pjm_hourly_est.csv"

COMBINED_LABEL = "PJM and ISO combined archive"

# Long-format master: every regional reading stacked into one frame,
# from which the Network Overview page is built.
MASTER_PARQUET = "all_hourly_long.parquet"

MASTER_LABEL = "All combined in one long archive"

REGION_COLUMN = "Region"

REGION_LABEL_COLUMN = "Region_Label"


# Cache so switching between pages does not re-read every archive
_CACHE = {}


# =========================================================
# REGION DISCOVERY
# =========================================================

def list_regions() -> list:
    """Return every hourly region archive found in the project folder."""

    regions = []

    for csv_file in sorted(DATA_DIR.glob("*_hourly.csv")):

        key = csv_file.stem

        regions.append(
            {
                "key": key,
                "label": REGION_LABELS.get(key, key),
                "file": csv_file.name,
                "is_default": key == DEFAULT_REGION,
            }
        )

    if not regions:

        raise FileNotFoundError(
            f"No hourly region archives found in: {DATA_DIR}"
        )

    return regions


def get_region_label(region: str) -> str:
    return REGION_LABELS.get(region, region)


def resolve_region(region: str = DEFAULT_REGION) -> str:
    """Validate the requested region and return a usable key."""

    if region is None or region == "":
        return DEFAULT_REGION

    available = [item["key"] for item in list_regions()]

    if region not in available:

        raise KeyError(
            f"Region '{region}' not found. Available regions: "
            + ", ".join(available)
        )

    return region


# =========================================================
# LOADING
# =========================================================

def load_data(region: str = DEFAULT_REGION) -> pd.DataFrame:
    """Read one region archive into a clean hourly time series."""

    region = resolve_region(region)

    if region in _CACHE:

        return _CACHE[region].copy()

    path = DATA_DIR / f"{region}.csv"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    if DATE_COLUMN not in df.columns:

        raise ValueError(
            f"'{DATE_COLUMN}' column missing from {path.name}"
        )

    value_column = df.columns[1]

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    df[TARGET] = pd.to_numeric(
        df[value_column],
        errors="coerce"
    )

    df = df[[DATE_COLUMN, TARGET]].copy()

    _CACHE[region] = df

    return df.copy()


def get_duplicate_count(region: str = DEFAULT_REGION) -> int:
    df = load_data(region)

    duplicate_count = df.duplicated().sum()

    return int(duplicate_count)


def get_value_column(region: str = DEFAULT_REGION) -> str:
    """Original demand column name as stored in the archive."""

    path = DATA_DIR / f"{region}.csv"

    df = pd.read_csv(path, nrows=0)

    return df.columns[1]


# =========================================================
# DATA SUMMARY
# =========================================================

def get_data_summary(region: str = DEFAULT_REGION) -> dict:

    region = resolve_region(region)

    df = load_data(region)

    timestamps = df[DATE_COLUMN]

    demand = df[TARGET]

    valid_demand = demand.dropna()

    # Expected hours between the first and last reading
    if timestamps.notna().any():

        span = timestamps.max() - timestamps.min()

        expected_hours = int(span.total_seconds() // 3600) + 1

    else:

        expected_hours = 0

    duplicate_timestamps = int(
        timestamps.duplicated().sum()
    )

    # Mean demand for each hour of the day
    hourly_profile = []

    if timestamps.notna().any():

        profile = (
            df.dropna(subset=[DATE_COLUMN])
            .groupby(timestamps.dt.hour)[TARGET]
            .mean()
        )

        for hour, mean_value in profile.items():

            hourly_profile.append(
                {
                    "hour": f"{int(hour):02d}:00",
                    "mean_demand": round(float(mean_value), 2),
                }
            )

    # Mean demand for each day of the week
    weekday_profile = []

    if timestamps.notna().any():

        weekdays = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        profile = (
            df.dropna(subset=[DATE_COLUMN])
            .groupby(timestamps.dt.dayofweek)[TARGET]
            .mean()
        )

        for day, mean_value in profile.items():

            weekday_profile.append(
                {
                    "weekday": weekdays[int(day)],
                    "mean_demand": round(float(mean_value), 2),
                }
            )

    # Longest stretch of hours with no reading at all
    gap_summary = {
        "missing_hours": 0,
        "longest_gap_hours": 0,
    }

    clean_timestamps = timestamps.dropna().sort_values()

    if len(clean_timestamps) > 1:

        deltas = clean_timestamps.diff().dt.total_seconds() / 3600

        missing_hours = (deltas[deltas > 1] - 1).sum()

        gap_summary = {
            "missing_hours": int(missing_hours) if missing_hours > 0 else 0,
            "longest_gap_hours": int(deltas.max() - 1)
            if deltas.max() > 1
            else 0,
        }

    summary = {
        "region": region,
        "region_label": get_region_label(region),
        "file": f"{region}.csv",
        "value_column": get_value_column(region),

        "n_rows": df.shape[0],
        "n_cols": raw_column_count(region),

        "columns": [DATE_COLUMN, get_value_column(region)],

        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        "missing_counts": {
            DATE_COLUMN: int(df[DATE_COLUMN].isna().sum()),
            TARGET: int(df[TARGET].isna().sum()),
        },

        "duplicate_count": get_duplicate_count(region),
        "duplicate_timestamps": duplicate_timestamps,

        "start_time": str(timestamps.min()),
        "end_time": str(timestamps.max()),

        "expected_hours": expected_hours,
        "missing_hours": gap_summary["missing_hours"],
        "longest_gap_hours": gap_summary["longest_gap_hours"],

        "numerical_stats": {
            TARGET: {
                "min": round(float(valid_demand.min()), 2),
                "max": round(float(valid_demand.max()), 2),
                "mean": round(float(valid_demand.mean()), 2),
                "std": round(float(valid_demand.std()), 2),
            }
        },

        "hourly_profile": hourly_profile,
        "weekday_profile": weekday_profile,

        "preview": df.head(10).to_dict(orient="records"),
    }

    return summary


def raw_column_count(region: str) -> int:
    path = DATA_DIR / f"{region}.csv"

    df = pd.read_csv(path, nrows=0)

    return df.shape[1]


def get_regions_overview() -> dict:
    """One row per region archive, used by the data loading page."""

    if "overview" in _CACHE:

        return _CACHE["overview"]

    rows = []

    for item in list_regions():

        path = DATA_DIR / item["file"]

        size_mb = path.stat().st_size / (1024 * 1024)

        header = pd.read_csv(path, nrows=0)

        rows.append(
            {
                "key": item["key"],
                "label": item["label"],
                "file": item["file"],
                "columns": header.shape[1],
                "value_column": header.columns[1],
                "size_mb": round(size_mb, 2),
            }
        )

    overview = {
        "regions": rows,
        "total_files": len(rows),
        "total_size_mb": round(sum(r["size_mb"] for r in rows), 2),
    }

    _CACHE["overview"] = overview

    return overview


# =========================================================
# COMBINED ARCHIVE
# =========================================================

def get_combined_source() -> dict:
    """Report which combined file is available and why."""

    parquet = DATA_DIR / COMBINED_PARQUET

    csv = DATA_DIR / COMBINED_CSV_FALLBACK

    if parquet.exists():

        return {
            "available": True,
            "file": COMBINED_PARQUET,
            "format": "parquet",
            "size_mb": round(parquet.stat().st_size / (1024 * 1024), 2),
            "note": "read with pyarrow",
        }

    if csv.exists():

        return {
            "available": True,
            "file": COMBINED_CSV_FALLBACK,
            "format": "csv",
            "size_mb": round(csv.stat().st_size / (1024 * 1024), 2),
            "note": "csv fallback, install pyarrow for the parquet file",
        }

    return {
        "available": False,
        "file": None,
        "format": None,
        "size_mb": 0.0,
        "note": "no combined archive found in the project folder",
    }


def load_combined() -> pd.DataFrame:
    """Read the wide archive holding every regional series at once."""

    if "combined" in _CACHE:

        return _CACHE["combined"].copy()

    source = get_combined_source()

    if not source["available"]:

        raise FileNotFoundError(
            f"Neither {COMBINED_PARQUET} nor {COMBINED_CSV_FALLBACK} "
            f"was found in: {DATA_DIR}"
        )

    path = DATA_DIR / source["file"]

    if source["format"] == "parquet":

        import pyarrow.parquet as pq

        df = pq.read_table(path).to_pandas()

        # The timestamp is stored as the frame index, not a column
        if not isinstance(df.index, pd.RangeIndex):

            df = df.reset_index()

    else:

        df = pd.read_csv(path)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    # The parquet stamps are UTC aware, the csv twin is already naive
    if df[DATE_COLUMN].dt.tz is not None:

        df[DATE_COLUMN] = df[DATE_COLUMN].dt.tz_localize(None)

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    _CACHE["combined"] = df

    return df.copy()


def get_combined_summary() -> dict:
    """Coverage of each regional column inside the wide archive."""

    df = load_combined()

    source = get_combined_source()

    total_rows = len(df)

    demand_columns = [c for c in df.columns if c != DATE_COLUMN]

    columns = []

    for column in demand_columns:

        values = pd.to_numeric(df[column], errors="coerce")

        valid = values.notna()

        valid_count = int(valid.sum())

        if valid_count:

            first_valid = df.loc[valid, DATE_COLUMN].min()
            last_valid = df.loc[valid, DATE_COLUMN].max()

            mean_value = float(values[valid].mean())

        else:

            first_valid = None
            last_valid = None

            mean_value = None

        columns.append(
            {
                "column": column,
                "valid_readings": valid_count,
                "null_readings": total_rows - valid_count,
                "coverage": round(100 * valid_count / total_rows, 1),
                "first_valid": str(first_valid) if first_valid is not None else "n/a",
                "last_valid": str(last_valid) if last_valid is not None else "n/a",
                "mean_demand": round(mean_value, 2) if mean_value is not None else None,
            }
        )

    return {
        "file": source["file"],
        "format": source["format"],
        "size_mb": source["size_mb"],
        "note": source["note"],

        "rows": total_rows,
        "n_cols": df.shape[1],
        "demand_columns": len(demand_columns),

        "start_time": str(df[DATE_COLUMN].min()),
        "end_time": str(df[DATE_COLUMN].max()),

        "duplicate_timestamps": int(df[DATE_COLUMN].duplicated().sum()),
        "missing_timestamps": int(df[DATE_COLUMN].isna().sum()),

        "columns": columns,
    }


def get_archives_overview() -> dict:
    """Both the per region archives and the combined wide archive."""

    overview = get_regions_overview()

    source = get_combined_source()

    overview["combined_file"] = source["file"]
    overview["combined_format"] = source["format"]
    overview["combined_size_mb"] = source["size_mb"]
    overview["combined_available"] = source["available"]

    master = get_master_summary()

    overview["master_file"] = master["file"]
    overview["master_format"] = master["format"]
    overview["master_rows"] = master["rows"]
    overview["master_size_mb"] = master["size_mb"]
    overview["master_available"] = master["available"]

    overview["total_size_mb"] = round(
        overview["total_size_mb"] + source["size_mb"] + master["size_mb"],
        2,
    )

    return overview


# =========================================================
# MASTER COMBINED ARCHIVE (LONG FORMAT)
# =========================================================

def build_master_archive(force: bool = False) -> pd.DataFrame:
    """Stack every regional archive into one long file and save it.

    The result is a single parquet frame holding region, label, timestamp
    and demand columns. Written once and reused on later requests.
    """

    if not force and MASTER_PARQUET in _CACHE:

        return _CACHE[MASTER_PARQUET].copy()

    path = DATA_DIR / MASTER_PARQUET

    if not force and path.exists():

        frame = pd.read_parquet(path)

        _CACHE[MASTER_PARQUET] = frame

        return frame.copy()

    frames = []

    for item in list_regions():

        region_frame = load_data(item["key"])

        frames.append(
            pd.DataFrame(
                {
                    REGION_COLUMN: item["key"],
                    REGION_LABEL_COLUMN: item["label"],
                    DATE_COLUMN: region_frame[DATE_COLUMN],
                    TARGET: region_frame[TARGET],
                }
            )
        )

    master = pd.concat(frames, ignore_index=True)

    master = (
        master
        .sort_values([REGION_COLUMN, DATE_COLUMN])
        .reset_index(drop=True)
    )

    master.to_parquet(path, index=False)

    _CACHE[MASTER_PARQUET] = master

    return master.copy()


def load_master() -> pd.DataFrame:
    """Read the combined long archive, building it on first use."""

    return build_master_archive()


def get_master_summary() -> dict:
    """Cheap metadata about the combined archive.

    Rows and the time span are derived from the per region frames already
    in cache, so this page level summary never has to read the whole file.
    """

    path = DATA_DIR / MASTER_PARQUET

    available = path.exists()

    starts = []
    ends = []
    total_rows = 0

    regional = []

    for item in list_regions():

        frame = load_data(item["key"])

        timestamps = frame[DATE_COLUMN]

        total_rows += int(len(frame))

        if timestamps.notna().any():

            starts.append(timestamps.min())
            ends.append(timestamps.max())

        regional.append(
            {
                "key": item["key"],
                "label": item["label"],
                "rows": int(len(frame)),
            }
        )

    if not starts:

        span_start = None
        span_end = None

    else:

        span_start = str(min(starts))
        span_end = str(max(ends))

    size_mb = round(path.stat().st_size / (1024 * 1024), 2) if available else 0.0

    return {
        "file": MASTER_PARQUET,
        "format": "parquet",
        "size_mb": size_mb,
        "available": available,
        "rows": total_rows,
        "regions": len(regional),
        "start": span_start,
        "end": span_end,
        "regional": regional,
    }


if __name__ == "__main__":

    import json

    print(json.dumps(get_archives_overview(), indent=2, default=str))
    print()
    print(json.dumps(get_combined_summary(), indent=2, default=str))

