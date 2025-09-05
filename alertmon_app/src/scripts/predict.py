import os
import argparse
import pandas as pd
import joblib
from prophet import Prophet
from alertmon.features import extract_features, FEATURES
from alertmon.forecast import load_forecast_models

def load_models(models_dir):
    clf_path = os.path.join(models_dir, "classifier", "random_forest.pkl")
    clf = joblib.load(clf_path)
    print(f"✅ Loaded classifier from {clf_path}")

    forecasts = load_forecast_models(os.path.join(models_dir, "forecast"))
    print(f"✅ Loaded {len(forecasts)} Prophet models")
    return clf, forecasts

def predict(csv_path, models_dir, interval=60):
    df = pd.read_csv(csv_path, parse_dates=["received_time"])
    df = extract_features(df)
    clf, forecasts = load_models(models_dir)

    # ---- Classification ----
    X = df[FEATURES]
    preds = clf.predict(X)
    df["predicted_fail"] = preds
    print("\n📊 Classification Results:")
    print(df[["received_time", "event_name", "status", "predicted_fail"]].head())

    # ---- Forecasting ----
    print("\n📈 Forecast Results:")
    for event, model in forecasts.items():
        future = model.make_future_dataframe(periods=interval, freq="min")
        forecast = model.predict(future)
        print(f"\n🔮 Event: {event}")
        print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to test CSV")
    parser.add_argument("--models", default="models", help="Directory with saved models")
    parser.add_argument("--interval", type=int, default=60, help="Forecast interval (minutes)")
    args = parser.parse_args()
    predict(args.csv, args.models, args.interval)