import argparse, os, time
import json
import random
from datetime import datetime, timedelta

# Import the monitor_certs module
from alertmon.monitor_certs import monitor_certs

def run(interval: int = 30):
    """
    Main monitoring loop that processes certificates.
    """
    certs_log_path = "alertmon_app/logs/certs_log.json"
    certs_config_path = "alertmon_app/data/certs_config.json"

    print(f"📡 Certificate Monitor watching {certs_config_path} every {interval}s")

    while True:
        try:
            monitor_certs(certs_config_path, certs_log_path)
        except Exception as e:
            print(f"An error occurred during certificate monitoring: {e}")
        
        time.sleep(interval)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=10, help="Polling interval (seconds)")
    a = ap.parse_args()
    run(a.interval)
