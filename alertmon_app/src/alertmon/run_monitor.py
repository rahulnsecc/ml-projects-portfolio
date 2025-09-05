import argparse, os, time
import pandas as pd
import joblib
from prophet import Prophet
from datetime import timedelta

from alertmon.features import extract_features, FEATURES
from alertmon.thresholds import load_thresholds
from alertmon.forecast import load_forecast_models

def load_classifier(models_dir: str):
    p = os.path.join(models_dir, "classifier", "random_forest.pkl")
    return joblib.load(p) if os.path.exists(p) else None

def predict_next_window(model: Prophet, periods=60, freq='min'):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    fc = model.predict(future)
    future_fc = fc.iloc[-periods:][['ds','yhat']].sort_values('yhat', ascending=False)
    return str(future_fc.iloc[0]['ds']) if len(future_fc) else ""

def run(source_csv: str, models_dir: str, interval: int = 30):
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/monitor_log.csv"

    clf = load_classifier(models_dir)
    forecasts = load_forecast_models(os.path.join(models_dir, "forecast"))
    thresholds = load_thresholds(os.path.join(models_dir, "thresholds.json"))

    print(f"📡 Monitor watching {source_csv} every {interval}s")

    while True:
        if not os.path.exists(source_csv):
            print(f"⚠️ File not found: {source_csv}")
            time.sleep(interval)
            continue

        try:
            df = pd.read_csv(source_csv, parse_dates=['received_time'])
            
            # Process all records from the source file
            latest = df.copy()
            latest = extract_features(latest)

            # Classification
            if clf is not None:
                X = latest[FEATURES]
                pred = clf.predict(X)
                latest['predicted'] = pred
            else:
                latest['predicted'] = 0

            # Anomaly flag using thresholds on delta_min
            latest['anomaly'] = latest.apply(
                lambda r: bool(r['delta_min'] > thresholds.get(str(r['event_name']), float('inf'))),
                axis=1
            )

            # Get the next_time forecast for each event
            next_times = {}
            unique_events = latest['event_name'].unique()
            for ev in unique_events:
                model = forecasts.get(ev)
                if model:
                    try:
                        next_times[ev] = predict_next_window(model, periods=60, freq='min')
                    except Exception:
                        next_times[ev] = "N/A"
                else:
                    next_times[ev] = "N/A"
            
            latest['next_time'] = latest['event_name'].map(next_times)

            # Persist the full processed log
            out_cols = ['received_time','event_name','status','anomaly','next_time', 'predicted']
            latest[out_cols].to_csv(log_path, index=False, mode='w')
            
            print(f"✅ Wrote {len(latest)} rows to {log_path}.")

        except Exception as e:
            print(f"An error occurred: {e}")

        time.sleep(interval)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to the source alerts CSV")
    ap.add_argument("--models", default="models", help="Directory with trained models")
    ap.add_argument("--interval", type=int, default=10, help="Polling interval (seconds)")
    a = ap.parse_args()
    run(a.source, a.models, a.interval)
