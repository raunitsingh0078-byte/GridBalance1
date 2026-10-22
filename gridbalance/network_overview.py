"""Network overview report built from the combined long archive.

This page is the answer to "what does the whole grid look like": every
region stacked into one dataset, compared side by side in four charts and
one table.

"""

import numpy as np
import pandas as pd

from eda_charts import (
    PALETTE,
    bar_chart,
    donut_chart,
    multiline_chart,
    value_heatmap,
)
from load_data import (
    DATE_COLUMN,
    TARGET,
    list_regions,
    load_data,
)

# One distinct Apple system colour per region
NETWORK_COLORS = [
    "#0A84FF",
    "#5E5CE6",
    "#BF5AF2",
    "#FF375F",
    "#FF9F0A",
    "#FFD60A",
    "#30D158",
    "#64D2FF",
    "#FF6482",
    "#32ADE6",
    "#AC8E68",
    "#A2845E",
]

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

CACHE = {}


def _region_frame(key: str) -> pd.DataFrame:
    """Per region load, cached in the loader already."""

    return load_data(key)


def get_network_report() -> dict:
    """Every number and chart for the Network Overview page."""

    if "report" in CACHE:

        return CACHE["report"]

    regions = []

    for item in list_regions():

        frame = _region_frame(item["key"])

        timestamps = frame[DATE_COLUMN]

        demand = frame[TARGET]

        regions.append(
            {
                "key": item["key"],
                "label": item["label"],
                "file": f"{item['key']}.csv",
                "rows": int(len(frame)),
                "start": str(timestamps.min())[:10],
                "end": str(timestamps.max())[:10],
                "mean": round(float(demand.mean()), 1),
                "min": round(float(demand.min()), 1),
                "max": round(float(demand.max()), 1),
                "std": round(float(demand.std()), 2),
            }
        )

    regions_sorted = sorted(regions, key=lambda r: r["mean"], reverse=True)

    # -----------------------------------------------------
    # Hourly profile per region (one line each)
    # -----------------------------------------------------

    profiles = []

    for item in list_regions():

        frame = _region_frame(item["key"])

        mean_by_hour = frame.groupby(frame[DATE_COLUMN].dt.hour)[TARGET].mean()

        row = []

        for hour in range(24):

            value = mean_by_hour.get(hour)

            row.append(float(value) if value is not None else np.nan)

        profiles.append(row)

    # -----------------------------------------------------
    # Mean demand per month and region (grid for the heatmap)
    # -----------------------------------------------------

    month_grid = []

    for item in list_regions():

        frame = _region_frame(item["key"])

        mean_by_month = frame.groupby(frame[DATE_COLUMN].dt.month)[TARGET].mean()

        month_grid.append(
            [
                round(float(mean_by_month.get(month)), 0)
                if mean_by_month.get(month) is not None
                else None
                for month in range(1, 13)
            ]
        )

    # -----------------------------------------------------
    # Global numbers
    # -----------------------------------------------------

    all_means = [r["mean"] for r in regions]

    total_rows = sum(r["rows"] for r in regions)

    total_energy_twh = 0.0

    for item in list_regions():

        frame = _region_frame(item["key"])

        total_energy_twh += float(frame[TARGET].sum())

    total_energy_twh = round(total_energy_twh / 1_000_000, 2)

    start_dates = [r["start"] for r in regions]
    end_dates = [r["end"] for r in regions]

    row_labels = [r["label"] for r in regions]

    region_rows = [r["rows"] for r in regions]

    total_rows_by_region = sum(region_rows)

    stats = {
        "region_count": len(regions),
        "total_rows": total_rows,
        "avg_rows_per_region": round(total_rows / max(len(regions), 1)),
        "span_start": min(start_dates),
        "span_end": max(end_dates),
        "average_demand_all": round(float(np.mean(all_means)), 1),
        "largest_region": regions_sorted[0]["label"],
        "largest_region_mean": regions_sorted[0]["mean"],
        "smallest_region": regions_sorted[-1]["label"],
        "smallest_region_mean": regions_sorted[-1]["mean"],
        "total_energy_twh": total_energy_twh,
    }

    charts = [
        {
            "key": "mean_by_region",
            "title": "Average Demand by Region",
            "subtitle": "Bar chart of the mean hourly load for each of the "
                        "12 regions, largest first",
            "kind": "Bar chart",
            "wide": False,
            "svg": bar_chart(
                [r["label"] for r in regions_sorted],
                [r["mean"] for r in regions_sorted],
                "Average demand by region",
                xlabel="Region",
                ylabel="Demand (MW)",
                colors=NETWORK_COLORS,
            ),
        },
        {
            "key": "share_of_readings",
            "title": "Share of Combined Readings by Region",
            "subtitle": "Donut chart of how many hourly readings each region "
                        "contributes to the one combined archive",
            "kind": "Pie chart",
            "wide": False,
            "svg": donut_chart(
                [r["label"] for r in regions],
                [float(r["rows"]) for r in regions],
                "Share of combined readings by region",
                colors=NETWORK_COLORS,
            ),
        },
        {
            "key": "hourly_profiles",
            "title": "Hourly Demand Profiles, All 12 Regions",
            "subtitle": "Line overlay of the mean load for every hour of the "
                        "day, one colour per region",
            "kind": "Line chart",
            "wide": True,
            "svg": multiline_chart(
                profiles,
                [item["label"] for item in list_regions()],
                [f"{hour:02d}:00" for hour in range(24)],
                "Hourly demand profiles across regions",
                colors=NETWORK_COLORS,
            ),
        },
        {
            "key": "month_region_heatmap",
            "title": "Mean Demand by Month and Region",
            "subtitle": "Heatmap of the average load across the twelve "
                        "calendar months for each region",
            "kind": "Heatmap",
            "wide": True,
            "svg": value_heatmap(
                month_grid,
                row_labels,
                MONTH_LABELS,
                "Mean demand by month and region",
                unit="MW",
            ),
        },
    ]

    report = {
        "stats": stats,
        "regions": regions_sorted,
        "charts": charts,
        "regional_rows": {
            "labels": [r["label"] for r in regions],
            "values": region_rows,
            "total": total_rows_by_region,
        },
    }

    CACHE["report"] = report

    return report


if __name__ == "__main__":

    import json

    result = get_network_report()

    stripped = dict(result)

    stripped["charts"] = [
        {k: (f"<{len(v)} chars of svg>" if k == "svg" else v)
         for k, v in chart.items()}
        for chart in result["charts"]
    ]

    print(json.dumps(stripped, indent=2, default=str))