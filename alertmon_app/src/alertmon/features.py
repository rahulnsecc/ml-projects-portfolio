
import pandas as pd

FEATURES = ['hour','dow','dom','month','delta_min']

def add_time_parts(df: pd.DataFrame, time_col: str = 'received_time') -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['dow'] = df[time_col].dt.dayofweek
    df['dom'] = df[time_col].dt.day
    df['month'] = df[time_col].dt.month
    return df

def compute_deltas(df: pd.DataFrame, time_col: str = 'received_time', event_col: str = 'event_name') -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([event_col, time_col])
    df['delta_min'] = (
        df.groupby(event_col)[time_col]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )
    return df

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_parts(df, 'received_time')
    df = compute_deltas(df, 'received_time', 'event_name')
    return df
