from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import datetime, timedelta
import os
import json

app = Flask(__name__)

DATA_FILE = os.environ.get("ALERTS_CSV", "data/alerts_history.csv")
CERTS_FILE = "logs/certs_log.json"

def load_df():
    df = pd.read_csv(DATA_FILE, parse_dates=["received_time"])
    # Normalize types
    df["event_name"] = df["event_name"].astype(str)
    df["status"] = df["status"].astype(str).str.upper()
    
    # Load additional columns for job details
    if 'from' in df.columns:
        df['from'] = df['from'].astype(str)
    if 'to' in df.columns:
        df['to'] = df['to'].astype(str)
    if 'subject' in df.columns:
        df['subject'] = df['subject'].astype(str)
    if 'source_path' in df.columns:
        df['source_path'] = df['source_path'].astype(str)
    if 'destination_path' in df.columns:
        df['destination_path'] = df['destination_path'].astype(str)
    if 'file_name' in df.columns:
        df['file_name'] = df['file_name'].astype(str)
    
    # Sort once by time desc for consistent pagination
    if "received_time" in df.columns:
        df = df.sort_values("received_time", ascending=False)
    return df

def load_certs():
    if not os.path.exists(CERTS_FILE):
        return []
    with open(CERTS_FILE, 'r') as f:
        return json.load(f)

def apply_filters(df, status, anomaly, time_filter, event, from_filter, to_filter):
    out = df.copy()

    # Time first
    if time_filter == "1d":
        cutoff = datetime.now() - timedelta(days=1)
        out = out[out["received_time"] >= cutoff]
    elif time_filter == "7d":
        cutoff = datetime.now() - timedelta(days=7)
        out = out[out["received_time"] >= cutoff]
    elif time_filter == "30d":
        cutoff = datetime.now() - timedelta(days=30)
        out = out[out["received_time"] >= cutoff]

    # Status
    if status.lower() != "all":
        out = out[out["status"].str.upper() == status.upper()]

    # Anomaly
    if anomaly.lower() == "true":
        out = out[out["anomaly"] == True]
    elif anomaly.lower() == "false":
        out = out[out["anomaly"] == False]

    # Event contains (case-insensitive)
    if event:
        needle = event.lower().strip()
        out = out[out["event_name"].str.lower().str.contains(needle, na=False)]
    
    # From contains (case-insensitive)
    if from_filter:
        needle = from_filter.lower().strip()
        out = out[out["from"].str.lower().str.contains(needle, na=False)]

    # To contains (case-insensitive)
    if to_filter:
        needle = to_filter.lower().strip()
        out = out[out["to"].str.lower().str.contains(needle, na=False)]

    return out

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    status = request.args.get("status", "all")
    anomaly = request.args.get("anomaly", "all")
    time_filter = request.args.get("time", "all")
    event = request.args.get("event", "")
    from_filter = request.args.get("from_filter", "")
    to_filter = request.args.get("to_filter", "")

    page = max(1, int(request.args.get("page", 1)))
    page_size = min(500, max(1, int(request.args.get("page_size", 50))))
    
    try:
        df = load_df()
        mon_df = pd.read_csv("logs/monitor_log.csv", parse_dates=["received_time", "next_time"])
        df = pd.merge(df, mon_df[['received_time', 'event_name', 'anomaly', 'next_time']], on=['received_time', 'event_name'], how='left')
    except FileNotFoundError:
        df = pd.DataFrame(columns=["received_time", "event_name", "status", "anomaly", "next_time", "from", "to"])
        df["received_time"] = pd.to_datetime(df["received_time"])
    
    df['anomaly'] = df['anomaly'].fillna(False)
    df['next_time'] = df['next_time'].astype(str).replace('NaT', '')

    out = apply_filters(df, status, anomaly, time_filter, event, from_filter, to_filter)
    out = out.sort_values("received_time", ascending=False)

    total = int(len(out))
    total_pages = max(1, (total + page_size - 1) // page_size)

    if page > total_pages:
        page = total_pages

    start = (page - 1) * page_size
    end = start + page_size
    page_df = out.iloc[start:end]

    fields = ["received_time", "event_name", "status", "anomaly", "next_time", "from", "to"]
    for f in fields:
        if f not in page_df.columns:
            page_df[f] = ""

    resp = page_df[fields].copy()
    resp["received_time"] = resp["received_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    resp["anomaly"] = resp["anomaly"].astype(str)

    return jsonify({
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "rows": resp.to_dict(orient="records")
    })

@app.route("/summary")
def summary():
    time_filter = request.args.get("time", "all")
    event = request.args.get("event", "")
    from_filter = request.args.get("from_filter", "")
    to_filter = request.args.get("to_filter", "")

    df = load_df()
    
    try:
        mon_df = pd.read_csv("logs/monitor_log.csv", parse_dates=["received_time"])
        df = pd.merge(df, mon_df[['received_time', 'event_name', 'anomaly']], on=['received_time', 'event_name'], how='left')
    except FileNotFoundError:
        df['anomaly'] = False
        
    df['anomaly'] = df['anomaly'].fillna(False)

    out = apply_filters(df, status="all", anomaly="all", time_filter=time_filter, event=event, from_filter=from_filter, to_filter=to_filter)

    total = int(len(out))
    success = int((out["status"].str.upper() == "SUCCESS").sum())
    failed = int((out["status"].str.upper() == "FAILED").sum())
    anomalies = int(out["anomaly"].sum())

    return jsonify({
        "total": total,
        "success": success,
        "failed": failed,
        "anomalies": anomalies,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
@app.route("/today_at_a_glance")
def today_at_a_glance():
    try:
        mon_df = pd.read_csv("logs/monitor_log.csv", parse_dates=["received_time", "next_time"])
    except FileNotFoundError:
        return jsonify([])

    latest_events = mon_df.sort_values('received_time', ascending=False).drop_duplicates('event_name').sort_values('event_name')

    status_list = []
    
    for _, row in latest_events.iterrows():
        event_name = row['event_name']
        received_time = row['received_time']
        last_run_status = row['status']
        next_run_forecast = "N/A"
        
        if pd.notna(row['next_time']):
            next_run_forecast = row['next_time'].strftime("%Y-%m-%d %H:%M:%S")
            
        status_list.append({
            'event_name': event_name,
            'last_run': received_time.strftime("%Y-%m-%d %H:%M:%S"),
            'last_run_status': last_run_status,
            'next_run_forecast': next_run_forecast
        })
    
    return jsonify(status_list)

@app.route("/event_history_details/<event_name>")
def event_history_details(event_name):
    page = max(1, int(request.args.get("page", 1)))
    page_size = min(500, max(1, int(request.args.get("page_size", 10))))
    time_filter = request.args.get("time", "all")
    status_filter = request.args.get("status", "all")
    from_filter = request.args.get("from_filter", "")
    to_filter = request.args.get("to_filter", "")


    df = load_df()
    df_event = df[df["event_name"] == event_name].copy()

    if time_filter == "1d":
        cutoff = datetime.now() - timedelta(days=1)
        df_event = df_event[df_event["received_time"] >= cutoff]
    elif time_filter == "7d":
        cutoff = datetime.now() - timedelta(days=7)
        df_event = df_event[df_event["received_time"] >= cutoff]
    elif time_filter == "30d":
        cutoff = datetime.now() - timedelta(days=30)
        df_event = df_event[df_event["received_time"] >= cutoff]

    if status_filter != "all":
        df_event = df_event[df_event["status"].str.upper() == status_filter.upper()]
    
    if from_filter:
        needle = from_filter.lower().strip()
        df_event = df_event[df_event["from"].str.lower().str.contains(needle, na=False)]
    
    if to_filter:
        needle = to_filter.lower().strip()
        df_event = df_event[df_event["to"].str.lower().str.contains(needle, na=False)]

    df_event = df_event.sort_values("received_time", ascending=False)
    
    if df_event.empty:
        return jsonify({ "rows": [], "total": 0, "total_pages": 0 })

    total = int(len(df_event))
    total_pages = max(1, (total + page_size - 1) // page_size)

    if page > total_pages:
        page = total_pages

    start = (page - 1) * page_size
    end = start + page_size
    page_df = df_event.iloc[start:end]

    display_cols = ['received_time', 'status', 'from', 'to', 'subject', 'source_path', 'destination_path', 'file_name']
    
    for col in display_cols:
        if col not in page_df.columns:
            page_df[col] = ""
    
    page_df = page_df[display_cols]
    page_df['received_time'] = page_df['received_time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return jsonify({
        "rows": page_df.to_dict(orient="records"),
        "total": total,
        "total_pages": total_pages,
        "page": page,
        "page_size": page_size
    })

@app.route("/certs")
def certs():
    cert_type = request.args.get("type", "all")
    status = request.args.get("status", "all")
    name = request.args.get("name", "")
    expiration_date_str = request.args.get("expiration_date", "")
    owner_team = request.args.get("owner_team", "")
    vendor_name = request.args.get("vendor_name", "")

    certs_data = load_certs()
    
    filtered_certs = []
    for cert in certs_data:
        # Apply filters
        if cert_type != "all" and cert.get('type') != cert_type:
            continue
        if status != "all" and cert.get('status') != status:
            continue
        if name and name.lower() not in cert.get('name', '').lower():
            continue
        if owner_team and owner_team.lower() not in cert.get('owner_team', '').lower():
            continue
        if vendor_name and vendor_name.lower() not in cert.get('vendor_name', '').lower():
            continue
        
        # New: Filter by expiration date
        if expiration_date_str:
            try:
                filter_date = datetime.strptime(expiration_date_str, '%Y-%m-%d').date()
                cert_exp_date = datetime.strptime(cert.get('expiration_date'), '%Y-%m-%d').date()
                if cert_exp_date > filter_date:
                    continue
            except (ValueError, TypeError):
                # Ignore invalid date filter
                pass
        
        filtered_certs.append(cert)
    
    return jsonify(filtered_certs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
