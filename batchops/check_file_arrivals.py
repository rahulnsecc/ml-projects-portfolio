# Filename: check_file_arrivals.py
import os
import pandas as pd
from datetime import datetime, timedelta, date, time
import fnmatch
import logging
import configparser
from pathlib import Path
import shutil
import csv
import glob # To find previous alert files for backup

# --- Configuration Loading ---
config = configparser.ConfigParser()
config_file = 'config.ini'
script_name = Path(__file__).stem
today_str = date.today().strftime('%Y-%m-%d') # Get today's date string early
current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Timestamp for this run

if not os.path.exists(config_file):
    print(f"FATAL: Configuration file '{config_file}' not found.")
    exit(1)

try:
    config.read(config_file)

    LOG_DIR = Path(config.get('Paths', 'log_dir', fallback='logs'))
    INPUT_DIR = Path(config.get('Paths', 'intermediate_dir', fallback='output'))
    OUTPUT_DIR = Path(config.get('Paths', 'output_dir', fallback='output'))

    CHECKER_LOG_BASE = config.get('Filenames', 'checker_log_base', fallback=script_name)
    PATTERNS_REPORT_CSV = config.get('Filenames', 'patterns_report_csv', fallback='file_patterns_report.csv')
    ARRIVAL_STATUS_CSV = config.get('Filenames', 'arrival_status_csv', fallback='file_arrival_status.csv')
    ALERT_CSV_BASE = config.get('Filenames', 'alert_csv_base', fallback='file_alerts') # Use BASE name

    SIZE_DEVIATION_THRESHOLD = config.getfloat('Alerting', 'size_deviation_threshold_percent', fallback=50.0)

    # --- Daily Log File Setup ---
    LOG_FILE = LOG_DIR / f"{CHECKER_LOG_BASE}_{today_str}.log"

    # --- File Paths ---
    INPUT_CSV_PATH = INPUT_DIR / PATTERNS_REPORT_CSV
    OUTPUT_STATUS_PATH = OUTPUT_DIR / ARRIVAL_STATUS_CSV # Non-daily status file
    ALERT_FILE_PATH_TODAY = OUTPUT_DIR / f"{ALERT_CSV_BASE}_{today_str}.csv" # Today's alert file

except (configparser.NoSectionError, configparser.NoOptionError, configparser.Error) as e:
    print(f"FATAL: Error reading configuration file '{config_file}': {e}")
    exit(1)

# --- Directory Creation ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Configure Logging (Appends to daily log) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent adding handlers multiple times if script is re-imported/run in same process
if not logger.handlers:
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # File Handler (Append)
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)


# --- Alert File Handling Functions ---

def backup_previous_days_alerts(output_dir: Path, alert_base_name: str, today_str: str):
    """Finds alert files from previous days and backs them up."""
    backup_count = 0
    logger.info("Checking for previous day's alert files to back up...")
    try:
        # Pattern to find dated alert files
        alert_pattern = str(output_dir / f"{alert_base_name}_????-??-??.csv")
        for alert_file in glob.glob(alert_pattern):
            alert_path = Path(alert_file)
            # Extract date from filename
            try:
                file_date_str = alert_path.stem.split('_')[-1]
                # Don't backup today's file or already backed up files
                if file_date_str != today_str and not str(alert_path).endswith('.bak'): # Check suffix properly
                    datetime.strptime(file_date_str, '%Y-%m-%d') # Validate date format
                    backup_path = alert_path.with_suffix('.csv.bak') # Simple .bak suffix
                    # Avoid re-backing up
                    if not backup_path.exists():
                        alert_path.rename(backup_path)
                        logger.info(f"Backed up previous alert file: '{alert_path.name}' to '{backup_path.name}'")
                        backup_count += 1
            except (IndexError, ValueError):
                 logger.warning(f"Could not parse date from potential alert file: {alert_path.name}")
            except OSError as e:
                 logger.error(f"Error backing up alert file {alert_path.name}: {e}")

    except Exception as e:
        logger.error(f"Error during alert file backup process: {e}", exc_info=True)
    if backup_count > 0:
        logger.info(f"Completed backup of {backup_count} previous day alert file(s).")
    else:
        logger.info("No previous day alert files found needing backup.")


def read_alerts_state(alert_file_path: Path) -> dict | None: # Return None on header error
    """
    Reads the current day's alert CSV into a dictionary keyed by alert tuple.
    Returns None if headers are incorrect, raises other IOErrors.
    Key: (AlertType, FolderPath, FilePrefix)
    Value: Dictionary representing the alert row.
    """
    alerts_state = {}
    if not alert_file_path.exists() or alert_file_path.stat().st_size == 0:
        logger.info(f"'{alert_file_path.name}' not found or empty. Starting with no existing alerts for today.")
        return alerts_state # Return empty dict if no file

    logger.info(f"Reading existing alert state from '{alert_file_path.name}'...")
    try:
        with open(alert_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            expected_cols = { # Expected columns for stateful alerts
                "FirstDetectedTimestamp", "LastDetectedTimestamp", "ResolutionTimestamp",
                "CurrentStatus", "AlertType", "Severity", "FolderPath", "FilePrefix", "Details"
            }
            # --- Check Headers ---
            if not expected_cols.issubset(reader.fieldnames or []):
                logger.error(f"Alert file '{alert_file_path.name}' has incorrect headers. Expected: {expected_cols}. Found: {reader.fieldnames}. Cannot process state.")
                return None # <<< Indicate header error by returning None
            # --- Headers OK, proceed reading ---
            for row_num, row in enumerate(reader, 1):
                try:
                    # Use the *original* AlertType for the key
                    key = (row['AlertType'], row['FolderPath'], row['FilePrefix'])
                    if key in alerts_state:
                        logger.warning(f"Duplicate alert key {key} found at row {row_num} in '{alert_file_path.name}'. Using first instance.")
                    else:
                        alerts_state[key] = row # Store the entire row dictionary
                except KeyError as e:
                    logger.warning(f"Skipping row {row_num} in '{alert_file_path.name}' due to missing key column: {e}. Row: {row}")
                    continue
        logger.info(f"Successfully read state for {len(alerts_state)} alerts from '{alert_file_path.name}'")
    except IOError as e: # Catch specific IO errors
         logger.error(f"IOError reading alerts state from {alert_file_path.name}: {e}", exc_info=True)
         raise # Reraise IOErrors as they likely prevent progress
    except Exception as e:
        logger.error(f"Unexpected error reading alerts state from {alert_file_path.name}: {e}", exc_info=True)
        raise # Reraise other unexpected errors

    return alerts_state


def rewrite_daily_alert_file(alert_file_path: Path, alerts_dict: dict):
    """Overwrites the daily alert file with the current state."""
    alerts_list = list(alerts_dict.values()) # Convert dict values back to list for writing
    header = [ # Define the exact header order
        "FirstDetectedTimestamp", "LastDetectedTimestamp", "ResolutionTimestamp",
        "CurrentStatus", "AlertType", "Severity", "FolderPath", "FilePrefix", "Details"
    ]

    if not alerts_list:
        # If no alerts, ensure the file is empty or doesn't exist
        if alert_file_path.exists():
            try:
                # Check size again before deleting, just in case
                if alert_file_path.stat().st_size == 0 or pd.read_csv(alert_file_path).empty: # Check if empty after potential header write
                     alert_file_path.unlink()
                     logger.info(f"Removed empty or header-only alert file: {alert_file_path.name}")
                # If it has content (somehow?), log a warning instead of deleting? Or maybe rewrite with header?
                # Let's stick to removing if it *should* be empty based on alerts_list
            except pd.errors.EmptyDataError: # Handle case where file exists but is truly empty
                 alert_file_path.unlink()
                 logger.info(f"Removed empty alert file: {alert_file_path.name}")
            except OSError as e:
                logger.error(f"Failed to remove potentially empty alert file {alert_file_path.name}: {e}")
        else:
            logger.info("No active alerts to write.")
        return

    # Proceed to write if there are alerts
    try:
        # Use 'w' mode to overwrite
        with open(alert_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header, extrasaction='ignore') # Ignore extra fields if any
            writer.writeheader()
            writer.writerows(alerts_list)
        logger.info(f"Successfully updated alert file '{alert_file_path.name}' with {len(alerts_list)} entries.")
    except IOError as e:
        logger.error(f"Failed to write alerts state to {alert_file_path.name}: {e}", exc_info=True)


# --- Helper Functions (time_str_to_dt_time, get_folder_files_details) ---
def time_str_to_dt_time(time_str, default_time=None):
    """Converts HH:MM string to datetime.time object, returns default on failure."""
    if pd.isna(time_str) or time_str in ["N/A", "Error", ""]:
        return default_time
    try:
        return datetime.strptime(time_str, "%H:%M").time()
    except (ValueError, TypeError):
        logger.warning(f"Invalid time format '{time_str}', using default: {default_time}")
        return default_time

def get_folder_files_details(folder_path_str, file_prefix):
    """
    Gets modification times and sizes (KB) for files matching a prefix in a folder.
    Returns a dictionary {filename: {'mtime': timestamp, 'size_kb': size}}.
    """
    folder_path = Path(folder_path_str)
    file_details = {}
    pattern = file_prefix if '*' in file_prefix or '?' in file_prefix else f"{file_prefix}*"

    if not folder_path.is_dir():
        logger.warning(f"Folder not found or not a directory: {folder_path}")
        return file_details

    try:
        for item in folder_path.iterdir():
            if item.is_file() and fnmatch.fnmatch(item.name, pattern):
                 try:
                     stat_result = item.stat()
                     file_details[item.name] = {
                         'mtime': stat_result.st_mtime,
                         'size_kb': stat_result.st_size / 1024.0
                     }
                 except (FileNotFoundError, PermissionError) as stat_err:
                      logger.warning(f"Could not stat file {item}: {stat_err}")
    except PermissionError as list_err:
        logger.error(f"Permission denied listing directory: {folder_path}: {list_err}")
    except Exception as e:
        logger.error(f"Error listing files in {folder_path} for prefix {file_prefix}: {e}", exc_info=False)

    return file_details

# --- Validation Function (Unchanged) ---
def validate_csv_row(row, index):
    """Validates a single row of the input patterns DataFrame."""
    errors = []
    required_columns = {"folder_path", "file_prefix", "frequency",
                        "max_creation_time_observed", "average_file_size_kb"}
    for col in required_columns:
        if col not in row:
             errors.append(f"Missing required column '{col}'")
        elif pd.isna(row[col]):
             # Allow N/A for time/size only if no data was ever found or single file
             if row.get('frequency') not in ['No Data', 'Single File', 'Irregular', 'Batched']:
                 errors.append(f"Empty value for required column '{col}'")

    valid_frequencies = {"Daily", "Weekly", "Monthly", "Single File", "Irregular", "Intra-day", "Batched", "No Data"}
    freq = row.get("frequency")
    if freq is not None and not pd.isna(freq) and freq not in valid_frequencies:
         errors.append(f"Invalid frequency '{freq}'. Valid values are: {', '.join(valid_frequencies)}")

    time_val = row.get('max_creation_time_observed')
    if time_val is not None and not pd.isna(time_val) and time_val not in ["N/A", "Error"]:
         if time_str_to_dt_time(time_val) is None:
             errors.append(f"Invalid time format '{time_val}'. Expected 'HH:MM' or 'N/A'.")

    avg_size = row.get('average_file_size_kb')
    if avg_size is not None and not pd.isna(avg_size) and avg_size != 'N/A':
        try:
            float(avg_size) # Check if convertible to float
        except (ValueError, TypeError):
             errors.append(f"Invalid average_file_size_kb '{avg_size}'. Expected number or 'N/A'.")


    if errors:
        logger.warning(f"Validation failed for input row {index} (Folder: {row.get('folder_path','?')}, Prefix: {row.get('file_prefix','?')}) : {'; '.join(errors)}")
        return False
    return True
# ---------------------------------------------------------------------------

# --- Main Checking Logic ---
def check_file_arrivals(input_csv_path, output_status_path, output_alert_path_today, alert_base_name, size_threshold):
    """
    Reads patterns, checks status, updates daily alert state, overwrites status report.
    """
    # --- Backup previous day's alerts (only if date changed) ---
    try:
        backup_previous_days_alerts(output_alert_path_today.parent, alert_base_name, today_str)
    except Exception as bk_err:
        logger.error(f"Error during alert backup: {bk_err}", exc_info=True)
        # Decide whether to continue if backup fails? For now, continue.

    # --- Read current alert state ---
    todays_alerts_state = None # Initialize
    reset_alert_file = False
    try:
        todays_alerts_state = read_alerts_state(output_alert_path_today)
        if todays_alerts_state is None: # Check for None return value (bad headers)
             reset_alert_file = True
             logger.warning(f"Incorrect headers detected in {output_alert_path_today.name}. Will backup and start fresh for today.")

    except Exception as read_err: # Catch errors raised by read_alerts_state (e.g., IOError)
        logger.critical(f"Halting execution because alert state could not be read from {output_alert_path_today.name}: {read_err}")
        return # Stop if we can't read the state

    # --- Handle incorrect headers by backing up and resetting state ---
    if reset_alert_file:
        logger.warning(f"Attempting to backup problematic alert file: {output_alert_path_today.name}")
        try:
            bad_header_backup_path = output_alert_path_today.with_suffix('.csv.badheader.bak')
            # Avoid overwriting previous bad backups from same day
            counter = 0
            while bad_header_backup_path.exists():
                counter += 1
                # Construct filename like file_alerts_DATE.csv.badheader1.bak
                base = output_alert_path_today.stem
                bad_header_backup_path = output_alert_path_today.with_name(f"{base}.badheader{counter}.bak")

            output_alert_path_today.rename(bad_header_backup_path)
            logger.info(f"Backed up alert file with bad headers to: {bad_header_backup_path.name}")
            todays_alerts_state = {} # Start with empty state for today
        except OSError as backup_err:
             logger.critical(f"Failed to back up alert file with bad headers ({output_alert_path_today.name}): {backup_err}. Halting execution.")
             return # Stop if we cannot backup the bad file
        except Exception as generic_backup_err:
             logger.critical(f"Unexpected error backing up alert file {output_alert_path_today.name}: {generic_backup_err}. Halting execution.")
             return

    # --- Read Input Patterns Report ---
    try:
        logger.info(f"Reading file patterns report from: {input_csv_path}")
        patterns_data = pd.read_csv(input_csv_path)
        logger.info(f"Successfully read {len(patterns_data)} rows from patterns report.")
    except FileNotFoundError:
        logger.error(f"Input patterns report file not found: {input_csv_path}. Cannot create status report.")
        # Write empty status report? No, only write if processing occurs.
        # But should we still write the (potentially empty) alert state? Yes.
        try:
             rewrite_daily_alert_file(output_alert_path_today, todays_alerts_state or {}) # Write current state even if no patterns
        except Exception as e:
             logger.error(f"Failed to write alert state file {output_alert_path_today.name} after pattern read error: {e}")
        return # Stop if input is missing
    except Exception as e:
        logger.error(f"Error reading patterns CSV file {input_csv_path}: {e}", exc_info=True)
        return # Stop on read error

    # --- Initialize Status Report ---
    status_report_rows = [] # Status report is always generated fresh

    # --- Process patterns and update alert state ---
    logger.info("Starting file arrival checks and alert state update...")
    processed_alert_keys_this_run = set() # Track keys touched in this run to avoid duplicate updates within one run

    for index, row in patterns_data.iterrows():

        if not validate_csv_row(row, index):
            status_report_rows.append({ # Add error row to status report
                "folder_path": row.get("folder_path", "N/A"),
                "file_prefix": row.get("file_prefix", "N/A"),
                "frequency": row.get("frequency", "Validation Error"),
                "expected_time": row.get("max_creation_time_observed", "N/A"),
                "status": "Error",
                "details": f"Invalid data in input row {index}. See checker log.",
                "latest_file_found_time": None,
                "latest_file_size_kb": None
            })
            continue # Skip processing this row

        # --- Extract validated data ---
        folder_path = str(row["folder_path"])
        file_prefix = row["file_prefix"]
        frequency = row["frequency"]
        expected_time_str = row["max_creation_time_observed"]
        avg_size_kb_hist = row["average_file_size_kb"]

        expected_dt_time = time_str_to_dt_time(expected_time_str, default_time=time.min)
        expected_datetime_today = datetime.combine(date.today(), expected_dt_time)

        # --- Get current file details ---
        current_file_details = get_folder_files_details(folder_path, file_prefix)
        latest_file_info = None
        latest_filename = None
        latest_file_dt = None
        latest_file_size_kb = None
        if current_file_details:
            try:
                latest_filename = max(current_file_details, key=lambda f: current_file_details[f]['mtime'])
                latest_file_info = current_file_details[latest_filename]
                latest_file_dt = datetime.fromtimestamp(latest_file_info['mtime'])
                latest_file_size_kb = latest_file_info['size_kb']
            except ValueError: # Handle empty current_file_details case after check
                 pass

        # --- Determine Status (Logic largely unchanged) ---
        status = "Unknown"
        details = ""
        is_missing = False
        file_found_today = bool(latest_file_dt and latest_file_dt.date() == date.today())

        if frequency == "No Data": status = "Info"; details = "Pattern analyzer found no historical files."
        elif frequency == "Single File":
            if latest_file_info: status = "OK"; details = f"Expected single/irregular file exists ({latest_filename})."
            else: status = "Alert"; details = "Expected single/irregular file not found."; is_missing = True
        elif frequency in ["Irregular", "Intra-day", "Batched"]:
             if latest_file_info: status = "OK"; details = f"File pattern has '{frequency}' frequency. Latest found: {latest_filename}."
             else: status = "Alert"; details = f"File pattern has '{frequency}' frequency, but no matching files found currently."; is_missing = True
        else: # Daily, Weekly, Monthly
            if file_found_today: status = "OK"; details = f"File found for today ({latest_file_dt.strftime('%Y-%m-%d %H:%M')})."
            elif datetime.now() > expected_datetime_today:
                status = "Alert"; details = f"File NOT found for today ({today_str}) and expected time ({expected_time_str if expected_time_str != 'N/A' else 'N/A'}) has passed."; is_missing = True
                if latest_file_dt: days_diff = (date.today() - latest_file_dt.date()).days; details += f" Last file found was on {latest_file_dt.date().isoformat()} ({days_diff} days ago)."
                else: details += " No previous file found matching pattern either."
            else: status = "Pending"; details = f"File not yet created for today ({today_str}). Expected around/by {expected_time_str if expected_time_str != 'N/A' else 'any time'}."
        # ---------------------------------------------------------------------------------

        # --- Update Alert State (In Memory) ---
        alert_key_missing = ('Missing File', folder_path, file_prefix)
        alert_key_size = ('Size Deviation', folder_path, file_prefix)
        alert_updated_or_created_this_row = False # Track if *any* alert action taken for this row

        # Check for Size Deviation first
        size_deviation_details = None
        if latest_file_info and avg_size_kb_hist != 'N/A' and size_threshold > 0:
             try:
                 historical_avg = float(avg_size_kb_hist)
                 if historical_avg > 0:
                     percent_diff = abs(latest_file_size_kb - historical_avg) / historical_avg * 100
                     if percent_diff > size_threshold:
                         size_deviation_details = f"File '{latest_filename}' size ({latest_file_size_kb:.2f} KB) deviates by {percent_diff:.1f}% from average ({historical_avg:.2f} KB). Threshold: {size_threshold}%."
             except (ValueError, TypeError): logger.warning(f"Invalid historical avg size '{avg_size_kb_hist}' for {file_prefix}")
             except Exception as size_ex: logger.error(f"Error checking size dev for {file_prefix}: {size_ex}")

        # Determine primary alert condition type for *this run*
        current_run_alert_type = None
        current_run_alert_details = details # Default details from status check
        current_run_severity = "Critical" # Default for missing

        if is_missing:
            current_run_alert_type = "Missing File"
        # Check size deviation condition *after* checking missing status
        if size_deviation_details:
            # If already missing, just append size info to missing details? Or separate alert?
            # Let's allow Size Deviation alert even if Missing File alert is also active/pending
            size_key = alert_key_size
            if size_key not in processed_alert_keys_this_run:
                if size_key in todays_alerts_state: # Update existing size alert
                    if todays_alerts_state[size_key]['CurrentStatus'] == 'Active':
                        todays_alerts_state[size_key]['LastDetectedTimestamp'] = current_timestamp
                        todays_alerts_state[size_key]['Details'] = size_deviation_details
                        processed_alert_keys_this_run.add(size_key)
                        alert_updated_or_created_this_row = True
                        logger.debug(f"Updated existing size alert for {size_key}")
                else: # Create new size alert
                     todays_alerts_state[size_key] = {
                        "FirstDetectedTimestamp": current_timestamp, "LastDetectedTimestamp": current_timestamp,
                        "ResolutionTimestamp": None, "CurrentStatus": 'Active', "AlertType": 'Size Deviation',
                        "Severity": 'Warning', "FolderPath": folder_path, "FilePrefix": file_prefix,
                        "Details": size_deviation_details}
                     processed_alert_keys_this_run.add(size_key)
                     alert_updated_or_created_this_row = True
                     logger.info(f"New alert generated: Size Deviation for {folder_path}/{file_prefix}")
            # Append size warning to main status details regardless
            details += f" [Size Deviation Warning]"


        # Handle Missing File alert state (if primary condition is missing)
        if current_run_alert_type == "Missing File":
            missing_key = alert_key_missing
            if missing_key not in processed_alert_keys_this_run:
                if missing_key in todays_alerts_state:
                    # Update existing missing alert
                    if todays_alerts_state[missing_key]['CurrentStatus'] == 'Active':
                        todays_alerts_state[missing_key]['LastDetectedTimestamp'] = current_timestamp
                        # Update details only if they've substantively changed (e.g., last found date changed)
                        # For simplicity, we'll update details every time for now.
                        todays_alerts_state[missing_key]['Details'] = current_run_alert_details
                        processed_alert_keys_this_run.add(missing_key)
                        alert_updated_or_created_this_row = True
                        logger.debug(f"Updated existing missing alert for {missing_key}")
                else:
                    # Create new missing alert
                    todays_alerts_state[missing_key] = {
                        "FirstDetectedTimestamp": current_timestamp, "LastDetectedTimestamp": current_timestamp,
                        "ResolutionTimestamp": None, "CurrentStatus": 'Active', "AlertType": 'Missing File',
                        "Severity": current_run_severity, "FolderPath": folder_path, "FilePrefix": file_prefix,
                        "Details": current_run_alert_details}
                    processed_alert_keys_this_run.add(missing_key)
                    alert_updated_or_created_this_row = True
                    logger.info(f"New alert generated: Missing File for {folder_path}/{file_prefix}")

        # Check for Resolution *after* potentially updating/creating missing alert state
        # Only resolve if the alert *was* active and is not being re-activated this run
        if status == 'OK' and alert_key_missing in todays_alerts_state and \
           todays_alerts_state[alert_key_missing]['CurrentStatus'] == 'Active' and \
           alert_key_missing not in processed_alert_keys_this_run: # Ensure not just created/updated

            resolution_details = f"Resolved: File '{latest_filename}' arrived at {latest_file_dt.strftime('%Y-%m-%d %H:%M:%S')}."
            todays_alerts_state[alert_key_missing]['CurrentStatus'] = 'Resolved'
            todays_alerts_state[alert_key_missing]['ResolutionTimestamp'] = current_timestamp
            todays_alerts_state[alert_key_missing]['Details'] = resolution_details
            # Set LastDetectedTimestamp to resolution time? Makes sense.
            todays_alerts_state[alert_key_missing]['LastDetectedTimestamp'] = current_timestamp
            processed_alert_keys_this_run.add(alert_key_missing) # Mark as processed
            alert_updated_or_created_this_row = True
            logger.info(f"Alert resolved: {alert_key_missing}")
            details += f" [Resolved at {current_timestamp}]" # Append to status


        # --- Append row to status report ---
        status_report_rows.append({
            "folder_path": folder_path, "file_prefix": file_prefix, "frequency": frequency,
            "expected_time": expected_time_str, "status": status, "details": details,
            "latest_file_found_time": latest_file_dt.strftime("%Y-%m-%d %H:%M:%S") if latest_file_dt else None,
            "latest_file_size_kb": f"{latest_file_size_kb:.2f}" if latest_file_size_kb is not None else None
        })
        # --- End of Loop ---

    # --- Write Status Report (Overwrite) ---
    try:
        report_df = pd.DataFrame(status_report_rows)
        report_df.to_csv(output_status_path, index=False, mode='w') # Overwrite
        logger.info(f"File arrival status report overwritten at {output_status_path}")
    except Exception as e:
        logger.error(f"Error writing arrival status report CSV {output_status_path}: {e}", exc_info=True)

    # --- Write Updated Alert State (Overwrite Daily File) ---
    try:
        rewrite_daily_alert_file(output_alert_path_today, todays_alerts_state)
    except Exception as e:
        logger.error(f"Failed to rewrite alert state file {output_alert_path_today.name}: {e}")


# --- Run the program ---
if __name__ == "__main__":
    start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"\n{'='*20} Starting File Arrival Check ({start_time_str}) {'='*20}")
    # Pass alert base name for backup function
    check_file_arrivals(INPUT_CSV_PATH, OUTPUT_STATUS_PATH, ALERT_FILE_PATH_TODAY, ALERT_CSV_BASE, SIZE_DEVIATION_THRESHOLD)
    end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{'='*20} File Arrival Check Completed ({end_time_str}) {'='*20}")