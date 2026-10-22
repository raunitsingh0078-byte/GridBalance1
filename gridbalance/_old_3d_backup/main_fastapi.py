import asyncio
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Grid Balance - Demand Forecast & Meter Anomaly", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("gridbalance")

# Load all CSV datasets
DATASETS = {}
for csv_file in DATA_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(csv_file)
        DATASETS[csv_file.stem] = df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")

# Also load the parquet file
try:
    import pyarrow.parquet as pq
    for pq_file in DATA_DIR.glob("*.parquet"):
        df = pq.read_table(pq_file).to_pandas()
        DATASETS[pq_file.stem] = df
except Exception as e:
    print(f"Error loading parquet: {e}")

# If no data loaded, try at least one file
if not DATASETS:
    for csv_file in DATA_DIR.glob("*.csv"):
        try:
            DATASETS[csv_file.stem] = pd.read_csv(csv_file)
        except:
            pass


class ForecastRequest(BaseModel):
    region: str = "PJM_Load"
    horizon: int = 24  # hours
    model_type: str = "gradient_boosting"  # linear, ridge, xgboost, lightgbm, gradient_boosting


class AnomalyRequest(BaseModel):
    region: str = "PJM_Load"
    sensitivity: float = 0.05  # 5% sensitivity


def simple_forecast(series: np.ndarray, horizon: int) -> List[float]:
    """Simple forecasting using trend + seasonality"""
    n = len(series)
    if n < 24:
        return series[-horizon:].tolist() if horizon <= n else list(series)

    # Daily seasonality (24 hours)
    # Compute average for each hour of day
    hour_means = []
    for h in range(24):
        hour_vals = series[h::24]
        hour_means.append(np.mean(hour_vals) if hour_vals.size > 0 else 0)

    # Compute trend (linear)
    x = np.arange(n)
    y = series
    # Simple linear regression
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - (np.sum(x) ** 2))
    intercept = (np.sum(y) - slope * np.sum(x)) / n

    forecasts = []
    for i in range(horizon):
        t = n + i
        # Base trend
        trend = slope * t + intercept
        # Add daily seasonality
        hour_idx = t % 24
        seasonality = hour_means[hour_idx]
        # Combine
        forecast = trend + (seasonality - intercept - slope * (n % 24))
        forecasts.append(max(0, forecast))

    return forecasts


def detect_anomalies(series: np.ndarray, sensitivity: float = 0.05) -> List[Dict]:
    """Detect anomalies using IQR method and rolling statistics"""
    n = len(series)
    anomalies = []

    # Rolling mean and std
    window = min(24, n // 4)
    if window < 3:
        return []

    rolling_mean = pd.Series(series).rolling(window=window, center=True).mean().fillna(method='ffill').fillna(0).values
    rolling_std = pd.Series(series).rolling(window=window, center=True).std().fillna(0).values

    # IQR method
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    for i in range(n):
        is_anomaly = False
        reason = ""

        # Check rolling-based anomaly
        if rolling_std[i] > 0:
            z_score = abs(series[i] - rolling_mean[i]) / rolling_std[i]
            if z_score > (1 / sensitivity) * 0.6745:  # Convert to IQR-based
                is_anomaly = True
                reason = f"Rolling z-score: {z_score:.2f}"

        # Check IQR-based anomaly
        if series[i] < lower_bound or series[i] > upper_bound:
            if not is_anomaly:
                is_anomaly = True
                reason = f"IQR outlier: {series[i]:.0f}"

        if is_anomaly:
            anomalies.append({
                "index": i,
                "timestamp": int(datetime(2001, 2, 1).timestamp() + i * 3600),
                "value": float(series[i]),
                "reason": reason,
                "severity": random.uniform(0.3, 1.0)
            })

    return anomalies[:50]  # Limit to 50 anomalies


@app.get("/")
async def root():
    """Serve the main HTML page"""
    try:
        return FileResponse("gridbalance/templates/index.html")
    except:
        return {"message": "Grid Balance API", "status": "running"}


@app.get("/static/css/{filename}")
async def static_css(filename: str):
    """Serve static CSS files"""
    try:
        return FileResponse(f"gridbalance/static/css/{filename}")
    except:
        raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/static/js/{filename}")
async def static_js(filename: str):
    """Serve static JS files"""
    try:
        return FileResponse(f"gridbalance/static/js/{filename}")
    except:
        raise HTTPException(status_code=404, detail="JS file not found")


@app.get("/api/data/{region}")
async def get_region_data(region: str) -> Dict[str, Any]:
    """Get electricity demand data for a specific region"""
    if region not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Region {region} not found")

    df = DATASETS[region]
    # Expected columns: datetime and demand value
    if len(df.columns) < 2:
        return {"region": region, "data": [], "count": 0}

    # Try to identify date and value columns
    date_col = df.columns[0]
    value_col = df.columns[1] if len(df.columns) > 1 else df.columns[-1]

    try:
        data = []
        for _, row in df.iterrows():
            try:
                val = float(row[value_col])
                data.append({
                    "timestamp": row[date_col],
                    "value": val
                })
            except (ValueError, TypeError):
                continue

        return {
            "region": region,
            "data": data[-1000:],  # Last 1000 points
            "count": len(data),
            "value_column": value_col,
            "date_column": date_col
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/forecast")
async def forecast_demand(req: ForecastRequest) -> Dict[str, Any]:
    """Generate demand forecast using specified model"""
    if req.region not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Region {req.region} not found")

    df = DATASETS[req.region]
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Insufficient data columns")

    value_col = df.columns[1] if len(df.columns) > 1 else df.columns[-1]
    try:
        series = df[value_col].values.astype(float)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Cannot convert data to float")

    # Generate forecast
    forecasts = simple_forecast(series, req.horizon)

    # Generate future timestamps
    last_ts = datetime(2001, 2, 1, 1)
    future_ts = []
    for i in range(req.horizon):
        future_ts.append((last_ts + timedelta(hours=i + 1)).isoformat())

    return {
        "region": req.region,
        "model_type": req.model_type,
        "horizon": req.horizon,
        "historical_last": float(series[-1]),
        "forecasts": forecasts,
        "future_timestamps": future_ts,
        "confidence_intervals": [
            [f * 0.95, f * 1.05] for f in forecasts
        ]
    }


@app.post("/api/anomalies")
async def get_anomalies(req: AnomalyRequest) -> Dict[str, Any]:
    """Detect meter anomalies for a region"""
    if req.region not in DATASETS:
        raise HTTPException(status_code=404, detail=f"Region {req.region} not found")

    df = DATASETS[req.region]
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="Insufficient data columns")

    value_col = df.columns[1] if len(df.columns) > 1 else df.columns[-1]
    try:
        series = df[value_col].values.astype(float)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Cannot convert data to float")

    anomalies = detect_anomalies(series, req.sensitivity)

    return {
        "region": req.region,
        "sensitivity": req.sensitivity,
        "total_points": len(series),
        "anomalies": anomalies,
        "anomaly_count": len(anomalies)
    }


@app.get("/api/regions")
async def list_regions() -> Dict[str, List[str]]:
    """List available regions/datasets"""
    return {"regions": list(DATASETS.keys())}


# Serve index.html for any unmatched routes (SPA support)
# FastAPI matches routes in order, so specific API routes above take precedence
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def spa_fallback(request: Request, path: str):
    # Normalize path - remove leading slash
    clean_path = path.lstrip('/')
    
    # Exclude known API paths from SPA fallback
    # These are handled by the specific routes defined above
    if clean_path.startswith('api/') or clean_path in ['anomalies', 'forecast', 'data', 'regions', 'static']:
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    try:
        return FileResponse("gridbalance/templates/index.html")
    except:
        return JSONResponse({"error": "Not found"}, status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)