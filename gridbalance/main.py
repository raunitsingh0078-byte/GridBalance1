"""Standalone walkthrough of the Grid Balance pipeline.

Run this file directly to see what each stage computes, with the
numbers printed step by step. The Flask application in App.py
serves exactly the same stages through the browser.
"""

import warnings

import numpy as np
import pandas as pd

from anomaly_detection import detect_anomalies, get_threshold
from load_data import DEFAULT_REGION, get_data_summary, list_regions, load_data
from model_training import train_models
from prediction import forecast
from preprocessing import clean_series, get_preprocessing_summary


warnings.filterwarnings("ignore")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

np.set_printoptions(suppress=True, precision=4)


REGION = DEFAULT_REGION


print("=" * 60)
print("GRID BALANCE PROJECT")
print("=" * 60)


# =========================================================
# DATA LOADING
# =========================================================

print("\n1. DATA LOADING")
print("-" * 60)

regions = list_regions()

print(f"archives found : {len(regions)}")
for item in regions:
    print(f"  {item['file']:<24} {item['label']}")

summary = get_data_summary(REGION)

print(f"\nregion     : {summary['region_label']} ({summary['region']})")
print(f"demand col : {summary['value_column']}")
print(f"shape      : {summary['n_rows']} rows x {summary['n_cols']} columns")
print(f"time span  : {summary['start_time']} -> {summary['end_time']}")
print(f"duplicates : {summary['duplicate_count']} rows, "
      f"{summary['duplicate_timestamps']} repeated timestamps")
print(f"gaps       : {summary['missing_hours']} missing hours, "
      f"longest {summary['longest_gap_hours']}")

stats = summary["numerical_stats"]["Demand_MW"]

print("\ndemand statistics")
print(f"  min  {stats['min']:>10,.2f} MW")
print(f"  max  {stats['max']:>10,.2f} MW")
print(f"  mean {stats['mean']:>10,.2f} MW")
print(f"  std  {stats['std']:>10,.2f} MW")


# =========================================================
# PREPROCESSING
# =========================================================

print("\n2. PREPROCESSING")
print("-" * 60)

prep = get_preprocessing_summary(REGION)

print(f"original rows        : {prep['original_rows']}")
print(f"repeated timestamps  : {prep['duplicate_timestamps_removed']}")
print(f"rows reordered       : {prep['rows_reordered']}")
print(f"missing hours filled : {prep['hours_filled']}")
print(f"outliers clipped     : {prep['outliers_clipped']}")
print(f"clip range           : {prep['lower_bound']} .. {prep['upper_bound']} MW")
print(f"processed rows       : {prep['processed_rows']}")
print(f"modelling rows       : {prep['modelling_rows']}")
print(f"features             : {prep['feature_count']}")

print("\nfeature groups")
groups = {}
for row in prep["feature_table"]:
    groups.setdefault(row["group"], []).append(row["feature"])
for group, features in groups.items():
    print(f"  {group:<20} {', '.join(features)}")

# Why one extra hour matters: the same wall clock hour on a weekday
# and on a weekend carries a completely different load
print("\nsame hour, different weekday")
profile = pd.DataFrame(summary["weekday_profile"])
print(profile.to_string(index=False))


# =========================================================
# FEATURE SANITY CHECK
# =========================================================

print("\n3. LEAKAGE CHECK")
print("-" * 60)

cleaned, _ = clean_series(load_data(REGION))

demand = cleaned["Demand_MW"]

previous_hour = demand.shift(1)
previous_week = demand.shift(168)

paired = pd.concat(
    [demand, previous_hour, previous_week], axis=1
).dropna()

paired.columns = ["Demand_MW", "Lag_1", "Lag_168"]

error_hour = (paired["Demand_MW"] - paired["Lag_1"]).abs().mean()
error_week = (paired["Demand_MW"] - paired["Lag_168"]).abs().mean()

print(f"mean |error| using lag 1   : {error_hour:>10,.2f} MW")
print(f"mean |error| using lag 168 : {error_week:>10,.2f} MW")
print(f"the previous hour explains the load better by "
      f"{100 * (1 - error_hour / error_week):.1f} percent")
print("both are kept, so the model sees short and weekly momentum")


# =========================================================
# MODEL TRAINING
# =========================================================

print("\n4. MODEL TRAINING")
print("-" * 60)

results = train_models(REGION)

print(f"training rows : {results['training_rows']}")
print(f"testing rows  : {results['testing_rows']}")
print(f"features      : {results['features']}")
print(f"test average  : {results['average_demand']} MW")

print("\nmodel                 MAE       RMSE        R2      MAPE%")
for name, metrics in results["results"].items():
    print(f"{name:<18} {metrics['mae']:>9,.2f} {metrics['rmse']:>10,.2f} "
          f"{metrics['r2']:>9,.4f} {metrics['mape']:>8,.2f}")

print("\nbaselines")
for name, metrics in results["baselines"].items():
    print(f"{name:<28} {metrics['mae']:>9,.2f} {metrics['rmse']:>10,.2f} "
          f"{metrics['r2']:>9,.4f} {metrics['mape']:>8,.2f}")

print(f"\nbest model : {results['best_model']}")
print(f"best RMSE  : {results['best_rmse']} MW")
print(f"best MAPE  : {results['best_mape']} percent")


# =========================================================
# FORECAST
# =========================================================

print("\n5. DEMAND FORECAST")
print("-" * 60)

rows = forecast(REGION, 24)

print(f"{'timestamp':<20} {'hour':<7} {'day':<10} {'predicted':>12} "
      f"{'lower':>12} {'upper':>12}")

for row in rows[:12]:
    print(f"{row['timestamp']:<20} {row['hour']:<7} {row['weekday']:<10} "
          f"{row['predicted_demand']:>12,.2f} {row['lower_bound']:>12,.2f} "
          f"{row['upper_bound']:>12,.2f}")

print(f"\npeak     {max(r['predicted_demand'] for r in rows):,.2f} MW")
print(f"low      {min(r['predicted_demand'] for r in rows):,.2f} MW")
print(f"average  {np.mean([r['predicted_demand'] for r in rows]):,.2f} MW")


# =========================================================
# ANOMALY DETECTION
# =========================================================

print("\n6. METER ANOMALY DETECTION")
print("-" * 60)

scan = detect_anomalies(REGION, 0.01)

print(f"window          : {scan['window_days']} days "
      f"({scan['total_points']} readings)")
print(f"robust z cut off: {scan['threshold']}")
print(f"normal theory   : {scan['reference_threshold']} (too tight here)")
print(f"anomalies       : {scan['anomaly_count']} "
      f"({scan['anomaly_rate']} percent)")
print(f"stuck meters    : {scan['stuck_readings']}")
print(f"impossible      : {scan['impossible_readings']}")

print("\nfindings by type")
for reason, count in scan["reason_counts"].items():
    print(f"  {count:>5}  {reason}")

print(f"\n{'timestamp':<20} {'reading':>12} {'expected':>12} {'z':>8}  reason")
for row in scan["rows"][:10]:
    print(f"{row['timestamp']:<20} {row['demand']:>12,.2f} "
          f"{row['expected']:>12,.2f} {row['z_score']:>8.2f}  {row['reason']}")

print("\nsensitivity sweep")
for sensitivity in [0.005, 0.01, 0.02, 0.05, 0.10]:
    result = detect_anomalies(REGION, sensitivity)
    print(f"  {sensitivity:>6.3f} -> z > {result['threshold']:>5.2f}  "
          f"{result['anomaly_count']:>4} flagged  "
          f"({result['anomaly_rate']} percent)")


print("\n" + "=" * 60)
print("Run App.py to explore these stages in the browser")
print("=" * 60)
