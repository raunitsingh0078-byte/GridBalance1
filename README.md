# Grid Balance Project

Hourly electricity load forecasting and anomaly detection for 12 PJM regions,
presented as a staged Flask dashboard.

## Requirements

- Python 3.13

## Setup

Install the dependencies into the project virtual environment:

```
python -m pip install -r requirements.txt
```

PyCharm is already configured to use `.venv` as the project interpreter.

## Running the app

From the `grpproject` folder:

```
python main.py
```

Then open <http://127.0.0.1:5005>.

Port notes: Grid Balance uses port **5005** by default so it never clashes
with the PlacementPrediction project, which listens on **5000**. To run
Grid Balance on another port:

```
set GRIDBALANCE_PORT=5001
python main.py
```

Then open <http://127.0.0.1:5001>.

To read the pipeline output in the console instead of the browser, run the
walkthrough script:

```
python gridbalance/main.py
```

If you run `gridbalance/App.py` directly in PyCharm, set the working directory
to `gridbalance` so Flask can find `templates/` and `static/`. The root
`main.py` handles this for you.

## Stages

| Route | Purpose |
| --- | --- |
| `/` | Hero overview, pipeline stages, and the 12 regional files |
| `/data-loading` | Per-region row counts, ranges, gaps, and both combined archives |
| `/preprocessing` | Cleaning results and 21 engineered features |
| `/eda-analysis` | Nine SVG charts: line, bar, histogram, box plot, scatter, heatmap, donut |
| `/network` | Cross-region charts from the combined long archive |
| `/model-training` | Four regressors benchmarked against mean and naive baselines |
| `/forecast` | Recursive hourly demand forecasts from a chosen start hour |
| `/anomaly-detection` | Robust-z outliers, stuck meters, and impossible readings |

Every stage has a region picker, so you can walk a single region through the
full pipeline.

## Datasets

The 12 regional CSVs (`*_hourly.csv`) live directly in `gridbalance/`. Two
combined files derive from them:

- `est_hourly.parquet` — the wide archive with every regional series side by
  side, loaded through `pyarrow` (falls back to `pjm_hourly_est.csv`).
- `all_hourly_long.parquet` — the long-format master: one row per region per
  hour (`Region`, `Region_Label`, `Datetime`, `Demand_MW`), built once and
  cached. It drives the `/network` page.

## Modules

| File | Responsibility |
| --- | --- |
| `gridbalance/App.py` | Flask app, routes, global stats context |
| `gridbalance/load_data.py` | Regional, combined, and long master archive loading |
| `gridbalance/eda_charts.py` | Inline SVG chart engine (line, bar, donut, heatmap, multi-line) |
| `gridbalance/network_overview.py` | Cross-region summary and charts for `/network` |
| `gridbalance/preprocessing.py` | Cleaning and feature engineering |
| `gridbalance/model_training.py` | Model fitting, baselines, metrics, benchmark cache |
| `gridbalance/prediction.py` | Recursive forecasting |
| `gridbalance/anomaly_detection.py` | Anomaly rules |
| `gridbalance/main.py` | Console walkthrough |
| `gridbalance/templates/index.html` | Single server-rendered template |
| `gridbalance/static/style.css` | Dashboard styling |

## Notes

- Model benchmarks are cached per region, so revisiting the training page for a
  region you already ran returns immediately.
- `gridbalance/_old_3d_backup/` holds the previous FastAPI and Three.js
  implementation. It is not used by the current app.