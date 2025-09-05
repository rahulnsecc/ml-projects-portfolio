import os
import joblib
import pandas as pd
from prophet import Prophet

def _hourly_counts_per_event(df: pd.DataFrame, event_col: str = 'event_name', time_col: str = 'received_time'):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    out = {}
    for ev, g in df.groupby(event_col):
        g = g.set_index(time_col).sort_index()
        hourly = g['status'].resample('60min').count().rename('y').to_frame()
        hourly['ds'] = hourly.index
        out[ev] = hourly[['ds','y']].reset_index(drop=True)
    return out

def train_forecast(df: pd.DataFrame, out_dir: str, event_col: str = 'event_name'):
    os.makedirs(out_dir, exist_ok=True)
    per_event = _hourly_counts_per_event(df, event_col=event_col, time_col='received_time')
    for ev, train_df in per_event.items():
        if len(train_df) < 10 or train_df['y'].sum() == 0:
            continue
        m = Prophet()
        m.fit(train_df)
        path = os.path.join(out_dir, f"{ev}_prophet.pkl")
        joblib.dump(m, path)
        print(f"✅ Saved Prophet model for {ev} → {path}")

def load_forecast_models(out_dir: str):
    models = {}
    if not os.path.exists(out_dir):
        return models
    for fname in os.listdir(out_dir):
        if fname.endswith('_prophet.pkl'):
            ev = fname.replace('_prophet.pkl','')
            try:
                models[ev] = joblib.load(os.path.join(out_dir, fname))
            except Exception as e:
                print(f"⚠️ Failed to load {fname}: {e}")
    return models