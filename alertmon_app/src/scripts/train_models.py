import os
import argparse
import pandas as pd
import joblib

from alertmon.features import extract_features
from alertmon.classifier import train_classifier
from alertmon.forecast import train_forecast
from alertmon.thresholds import compute_thresholds

def train_models(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=["received_time"])

    # Feature enrichment
    df_feat = extract_features(df)

    # Train classifier
    clf_dir = os.path.join(out_dir, "classifier")
    os.makedirs(clf_dir, exist_ok=True)
    clf_path = os.path.join(clf_dir, "random_forest.pkl")
    train_classifier(df_feat, clf_path)
    print(f"✅ Saved classifier → {clf_path}")

    # Train forecasting models
    fc_dir = os.path.join(out_dir, "forecast")
    os.makedirs(fc_dir, exist_ok=True)
    train_forecast(df_feat, fc_dir, event_col="event_name")

    # Compute thresholds for anomaly detection
    th_path = os.path.join(out_dir, "thresholds.json")
    compute_thresholds(df_feat, th_path)
    print(f"✅ Saved thresholds → {th_path}")

    print(f"🎉 All models trained and saved in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to training data CSV")
    parser.add_argument("--out", default="models", help="Directory to save models")
    args = parser.parse_args()
    train_models(args.csv, args.out)