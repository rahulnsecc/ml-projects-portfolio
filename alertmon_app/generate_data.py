import pandas as pd, numpy as np
from datetime import datetime, timedelta
import argparse, random, os

def generate(start, end, out):
    start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
    cur = start_dt
    rows=[]
    events = ["JobA","JobB","JobC","JobD"]
    freqs = {"JobA":240,"JobB":180,"JobC":1440,"JobD":10080}  # minutes between expected runs (roughly)
    while cur < end_dt:
        for ev, f in freqs.items():
            # 90% chance the alert comes in this hour
            if random.random()<0.9:
                rows.append({
                    "received_time": cur.strftime("%Y-%m-%d %H:%M:%S"),
                    "from":"noreply@sys.com",
                    "to":"ops@team.com",
                    "subject":f"Alert {ev}",
                    "event_name":ev,
                    "source_path":"/src",
                    "destination_path":"/dst",
                    "file_name":f"{ev}.csv",
                    "status": random.choice(["SUCCESS","FAILED"] if random.random()<0.15 else ["SUCCESS"])
                })
        cur += timedelta(minutes=60)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"✅ Wrote {len(rows)} rows to {out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out",default="data/alerts_history.csv")
    a=ap.parse_args()
    generate(a.start,a.end,a.out)
