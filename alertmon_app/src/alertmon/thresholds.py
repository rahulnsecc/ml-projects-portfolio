
import json
from pathlib import Path
import pandas as pd

def compute_thresholds(df: pd.DataFrame, out_path: str):
    thresholds = {}
    for event, grp in df.groupby('event_name'):
        deltas = grp['delta_min'][grp['delta_min'] > 0]
        if not deltas.empty:
            thresholds[event] = float(deltas.quantile(0.95))
    Path(out_path).write_text(json.dumps(thresholds, indent=2))
    return thresholds

def load_thresholds(path: str):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}
