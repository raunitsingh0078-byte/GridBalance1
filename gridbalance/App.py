import os

from flask import Flask, render_template, request

from anomaly_detection import (
    detect_anomalies,
    get_anomaly_form_options,
)
from eda_charts import get_eda_report
from load_data import (
    DEFAULT_REGION,
    get_archives_overview,
    get_combined_summary,
    get_data_summary,
    list_regions,
    resolve_region,
)
from model_training import train_models
from network_overview import NETWORK_COLORS, get_network_report
from prediction import get_forecast_form_options, predict_demand
from preprocessing import get_preprocessing_summary


app = Flask(__name__)


# =========================================================
# JINJA HELPERS
# =========================================================

@app.template_filter("el_region_colour")
def el_region_colour(index: int) -> str:
    """Row accent colour for the region comparison table."""

    return NETWORK_COLORS[int(index) % len(NETWORK_COLORS)]


# =========================================================
# GLOBAL CONTEXT
# =========================================================

@app.context_processor
def inject_global_stats():
    """Small site-wide facts so the topbar, hero and footer stay live."""

    overview = get_archives_overview()

    return {
        "site_stats": {
            "region_count": overview["total_files"],
            "combined_available": bool(overview["combined_available"]),
            "combined_file": overview["combined_file"],
            "master_available": bool(overview["master_available"]),
            "master_file": overview["master_file"],
            "master_rows": overview["master_rows"],
            "total_size_mb": overview["total_size_mb"],
        }
    }


# =========================================================
# SHARED HELPERS
# =========================================================

def selected_region(source) -> str:
    """Read the region from a form or a query string, never raising."""

    requested = None

    if source is not None:

        requested = source.get("region")

    try:

        return resolve_region(requested)

    except (KeyError, FileNotFoundError):

        return DEFAULT_REGION


def region_choices(selected: str) -> list:
    """Region list with the chosen entry marked, for the picker."""

    choices = []

    for item in list_regions():

        choices.append(
            {
                "key": item["key"],
                "label": item["label"],
                "file": item["file"],
                "selected": item["key"] == selected,
            }
        )

    return choices


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none",
        regions=region_choices(DEFAULT_REGION),
        overview=get_archives_overview()
    )


# =========================================================
# DATA LOADING
# =========================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None
    combined_summary = None

    region = selected_region(request.args)

    try:

        summary = get_data_summary(region)

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    try:

        combined_summary = get_combined_summary()

    except Exception:

        # The wide archive is optional, the page still works without it
        combined_summary = None

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        combined_summary=combined_summary,
        region=region,
        regions=region_choices(region),
        overview=get_archives_overview(),
        error=error
    )


# =========================================================
# PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    error = None
    preprocessing_summary = None

    region = selected_region(request.args)

    try:

        preprocessing_summary = get_preprocessing_summary(region)

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="preprocessing",
        preprocessing_summary=preprocessing_summary,
        region=region,
        regions=region_choices(region),
        error=error
    )


# =========================================================
# EDA & VISUALISATION
# =========================================================

@app.route("/eda-analysis")
def eda_analysis():

    error = None
    report = None

    region = selected_region(request.args)

    try:

        report = get_eda_report(region)

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="eda-analysis",
        eda_report=report,
        region=region,
        regions=region_choices(region),
        error=error
    )


# =========================================================
# NETWORK OVERVIEW
# =========================================================

@app.route("/network")
def network_overview():

    error = None
    report = None

    try:

        report = get_network_report()

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="network",
        network_report=report,
        region=DEFAULT_REGION,
        regions=region_choices(DEFAULT_REGION),
        error=error
    )


# =========================================================
# MODEL TRAINING
# =========================================================

@app.route("/model-training")
def model_training():

    error = None
    results = None

    region = selected_region(request.args)

    try:

        results = train_models(region)

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="model-training",
        results=results,
        region=region,
        regions=region_choices(region),
        error=error
    )


# =========================================================
# FORECAST
# =========================================================

@app.route("/forecast", methods=["GET", "POST"])
def forecast():

    forecast_result = None
    error = None

    options = get_forecast_form_options()

    region = DEFAULT_REGION

    if request.method == "POST":

        region = selected_region(request.form)

        try:

            forecast_result = predict_demand(request.form)

        except Exception as e:

            error = f"Forecast error: {e}"

    else:

        region = selected_region(request.args)

    return render_template(
        "index.html",
        active="forecast",
        forecast_result=forecast_result,
        options=options,
        region=region,
        regions=region_choices(region),
        error=error
    )


# =========================================================
# ANOMALY DETECTION
# =========================================================

@app.route("/anomaly-detection", methods=["GET", "POST"])
def anomaly_detection():

    anomaly_result = None
    error = None

    options = get_anomaly_form_options()

    region = DEFAULT_REGION

    if request.method == "POST":

        region = selected_region(request.form)

        try:

            sensitivity = float(request.form.get("sensitivity", 0.01))

        except (TypeError, ValueError):

            sensitivity = options["default_sensitivity"]

        try:

            anomaly_result = detect_anomalies(region, sensitivity)

        except Exception as e:

            error = f"Detection error: {e}"

    else:

        region = selected_region(request.args)

    return render_template(
        "index.html",
        active="anomaly-detection",
        anomaly_result=anomaly_result,
        options=options,
        region=region,
        regions=region_choices(region),
        error=error
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("GRIDBALANCE_PORT", 5005))

    app.run(debug=True, port=port)
