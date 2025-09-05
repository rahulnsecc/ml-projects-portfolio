# Filename: analyze_file_patterns.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from collections import Counter
import re
import logging
import configparser
from pathlib import Path
import shutil # For file backup

# --- Configuration Loading ---
config = configparser.ConfigParser()
config_file = 'config.ini'
script_name = Path(__file__).stem # Get the name of the current script for logging

if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

try:
    config.read(config_file)

    LOG_DIR = Path(config.get('Paths', 'log_dir', fallback='logs'))
    # Patterns report goes to intermediate_dir (which might be same as output_dir)
    OUTPUT_DIR = Path(config.get('Paths', 'intermediate_dir', fallback='output'))

    ANALYZER_LOG_BASE = config.get('Filenames', 'analyzer_log_base', fallback=script_name)
    PATTERNS_REPORT_CSV = config.get('Filenames', 'patterns_report_csv', fallback='file_patterns_report.csv')

    ROOT_FOLDERS_STR = config.get('Monitoring', 'root_folders', fallback='./test_environment/after_batch_run')
    ROOT_FOLDERS = [Path(p.strip()) for p in ROOT_FOLDERS_STR.split(',') if p.strip()]

    # --- Daily Log File Setup ---
    today_str = date.today().strftime('%Y-%m-%d')
    LOG_FILE = LOG_DIR / f"{ANALYZER_LOG_BASE}_{today_str}.log"

    # --- Output File Path ---
    OUTPUT_FILE_PATH = OUTPUT_DIR / PATTERNS_REPORT_CSV

except (configparser.NoSectionError, configparser.NoOptionError, configparser.Error) as e:
    print(f"Error reading configuration file '{config_file}': {e}")
    # Cannot log here as logging depends on config
    exit(1)

# --- Directory Creation ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Configure Logging (Appends to daily log) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'), # Ensure append mode for daily log
        logging.StreamHandler()
    ],
    # Force configuration even if already configured in the same process run
    force=True
)

# --- Backup Function ---
def backup_file_if_exists(file_path: Path):
    """Renames the file with a timestamp if it exists."""
    if file_path.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = file_path.with_suffix(f'_{timestamp}.bak')
        try:
            shutil.move(str(file_path), str(backup_path)) # Use shutil.move for cross-fs compatibility
            logging.info(f"Backed up existing file '{file_path}' to '{backup_path}'")
        except Exception as e:
            logging.error(f"Failed to backup file '{file_path}': {e}", exc_info=True)
            # Decide if you want to proceed without backup or stop
            # For now, we'll log the error and continue

# --- Core Functions (Mostly unchanged from previous refactoring) ---
# normalize_filename, remove_outliers, get_file_groups,
# most_common_times, calculate_frequency_with_clustering,
# calculate_max_creation_time

# (Include the definitions of the functions:
# normalize_filename, remove_outliers, get_file_groups, most_common_times,
# calculate_frequency_with_clustering, calculate_max_creation_time
# from the previous answer here, no changes needed in their core logic)
# --- Paste the function definitions here ---
def normalize_filename(filename):
    """
    Normalizes the filename by replacing numbers or dates just before the file extension with '*'.
    """
    base, ext = os.path.splitext(filename)
    # Improved regex: matches one or more digits optionally preceded/followed by common separators
    # Handles cases like file_20230101.txt, file-123.csv, file123.dat
    normalized = re.sub(r'[_-]?\d+$', '*', base) # Replace trailing numbers/dates with '*'
    return normalized + ext

def remove_outliers(data):
    """
    Removes outliers from a list of numeric data using the IQR method.
    """
    if len(data) < 4: # Need at least 4 points for meaningful IQR
        return data

    try:
        # Ensure data is numeric, handle potential strings or Nones
        numeric_data = [x for x in data if isinstance(x, (int, float))]
        if len(numeric_data) < 4:
             return numeric_data # Not enough numeric points

        q1 = np.percentile(numeric_data, 25)
        q3 = np.percentile(numeric_data, 75)
        iqr = q3 - q1
        # Handle cases where iqr is zero
        if iqr == 0:
            # If IQR is 0, all points are the same (within float precision perhaps)
            # Or data might be skewed. Return original numeric data.
            return numeric_data
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [x for x in numeric_data if lower_bound <= x <= upper_bound]
    except Exception as e:
        logging.warning(f"Could not calculate outliers, returning original numeric data. Error: {e}")
        return [x for x in data if isinstance(x, (int, float))] # Return only numeric part


def get_file_groups(root_folders_list):
    """
    Recursively scans the list of root folders for files and groups them by
    folder path (absolute) and normalized file prefix.
    """
    file_groups = {}
    logging.info(f"Scanning root folders: {', '.join(map(str, root_folders_list))}")

    for root_folder in root_folders_list:
        if not root_folder.is_dir():
            logging.warning(f"Root folder '{root_folder}' does not exist or is not a directory. Skipping.")
            continue

        for dirpath, _, filenames in os.walk(root_folder):
            current_dir = Path(dirpath)
            for filename in filenames:
                try:
                    full_path = current_dir / filename
                    if not full_path.is_file(): # Skip directories/symlinks etc.
                        continue

                    # Normalize the filename
                    normalized_name = normalize_filename(filename)
                    # More robust prefix extraction - split only if '*' is present
                    prefix = normalized_name.split(".*")[0] if '.*' in normalized_name else os.path.splitext(normalized_name)[0]

                    # Use absolute path for uniqueness across different roots
                    group_key = (str(current_dir.resolve()), prefix) # Use resolved absolute path

                    # Add file to the corresponding group
                    if group_key not in file_groups:
                        file_groups[group_key] = {"timestamps": [], "sizes": []}

                    stat_result = full_path.stat()
                    creation_time = stat_result.st_mtime # Modification time
                    file_size = stat_result.st_size / 1024 # File size in KB

                    file_groups[group_key]["timestamps"].append(creation_time)
                    file_groups[group_key]["sizes"].append(file_size)

                except (FileNotFoundError, PermissionError) as e:
                    logging.warning(f"Error accessing file: {full_path}. Skipping. Error: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error processing file: {full_path}. Error: {e}", exc_info=False) # Set exc_info=False for less verbose logs unless debugging


    logging.info(f"Found {len(file_groups)} distinct file groups.")
    return file_groups

def most_common_times(timestamps):
    """
    Finds the most common times in HH:MM format, grouping minutes into 30-minute intervals.
    Returns the top 3 most common times in a day.
    """
    if not timestamps:
        return []
    try:
        times = [datetime.fromtimestamp(ts).strftime("%H:%M") for ts in timestamps]
        # Group by 30-minute intervals (00 or 30)
        grouped_times = [f"{t.split(':')[0]}:{int(t.split(':')[1])//30*30:02d}" for t in times]
        most_common = Counter(grouped_times).most_common(3) # Get top 3 common times
        return [time for time, _ in most_common]
    except Exception as e:
        logging.warning(f"Could not determine most common times: {e}")
        return ["Error"]


def calculate_frequency_with_clustering(timestamps):
    """
    Calculates file creation frequency using simple interval clustering.
    """
    fallback_time = None
    avg_interval = None
    num_timestamps = len(timestamps)

    if num_timestamps == 0:
        return "No Data", fallback_time, avg_interval
    elif num_timestamps == 1:
        fallback_time = datetime.fromtimestamp(timestamps[0]).strftime("%H:%M")
        return "Single File", fallback_time, avg_interval

    timestamps.sort()
    # Calculate intervals in hours
    intervals = np.diff(timestamps) / 3600.0

    if not intervals.size: # Should not happen if num_timestamps > 1, but safety check
        fallback_time = datetime.fromtimestamp(timestamps[-1]).strftime("%H:%M")
        return "Single File", fallback_time, avg_interval

    # Calculate average on non-zero intervals if possible, avoids issues with files created simultaneously
    positive_intervals = intervals[intervals > 0]
    avg_interval = np.mean(positive_intervals) if positive_intervals.size > 0 else 0

    # Define cluster boundaries (in hours) - adjusted slightly
    daily_cluster = [i for i in intervals if 18 <= i <= 30]
    weekly_cluster = [i for i in intervals if 150 <= i <= 180] # ~6.25 to 7.5 days
    monthly_cluster = [i for i in intervals if 680 <= i <= 760] # ~28 to 31.6 days

    # Determine frequency based on the largest cluster (simple approach)
    len_d, len_w, len_m = len(daily_cluster), len(weekly_cluster), len(monthly_cluster)

    # Require a minimum number of intervals to be confident, e.g., 2 or 3? Let's use 2.
    min_intervals_for_freq = 2

    if len_d >= min_intervals_for_freq and len_d >= len_w and len_d >= len_m:
        frequency = "Daily"
    elif len_w >= min_intervals_for_freq and len_w >= len_m:
        frequency = "Weekly"
    elif len_m >= min_intervals_for_freq:
        frequency = "Monthly"
    else:
        # If not enough evidence for periodic, check if intervals are very short (e.g. hourly)
        if avg_interval > 0 and avg_interval < 6: # Check avg_interval > 0
             frequency = "Intra-day"
        elif avg_interval == 0 and num_timestamps > 1:
             # Multiple files, zero interval -> likely simultaneous or very close
             frequency = "Batched" # A new category
        else:
             frequency = "Irregular" # Default if no clear pattern or too few samples

        # Use the most recent time as a fallback if not Daily/Weekly/Monthly
        fallback_time = datetime.fromtimestamp(timestamps[-1]).strftime("%H:%M")


    return frequency, fallback_time, avg_interval


def calculate_max_creation_time(timestamps):
    """
    Calculates the latest time of day (HH:MM) observed among the timestamps.
    """
    if not timestamps:
        return "N/A"

    latest_time_obj = None
    try:
        for ts in timestamps:
          dt_obj = datetime.fromtimestamp(ts)
          current_time_part = dt_obj.time()
          if latest_time_obj is None or current_time_part > latest_time_obj:
              latest_time_obj = current_time_part

        return latest_time_obj.strftime("%H:%M") if latest_time_obj else "N/A"
    except Exception as e:
        logging.warning(f"Could not determine latest creation time: {e}")
        return "Error"

# ------------------------------------------------------------------------


def analyze_file_history(root_folders_list, output_csv_path):
    """
    Analyzes file creation history in the specified root folders and saves the results to a CSV file.
    """
    try:
        logging.info("Starting file group analysis...")
        file_groups = get_file_groups(root_folders_list)
        results = []

        if not file_groups:
            logging.warning("No files found or processed in the specified folders.")
            df = pd.DataFrame(columns=[
                 "folder_path", "file_prefix", "frequency", "fallback_time",
                 "average_interval_hours", "most_common_times", "max_creation_time_observed",
                 "average_file_size_kb", "last_seen"
            ])
            # Backup before writing even an empty file
            backup_file_if_exists(output_csv_path)
            df.to_csv(output_csv_path, index=False)
            logging.info(f"Empty file patterns report saved to {output_csv_path}")
            return

        logging.info(f"Analyzing {len(file_groups)} file groups...")
        for group_key, data in file_groups.items():
            dirpath, prefix = group_key
            timestamps = data["timestamps"]
            sizes = data["sizes"] # sizes are in KB

            # Remove outliers from sizes and calculate average size
            filtered_sizes = remove_outliers(sizes) # Expects KB
            avg_size_kb = np.mean(filtered_sizes) if filtered_sizes else "N/A"

            # Analyze timestamps
            common_times = most_common_times(timestamps)
            frequency, fallback_time, avg_interval = calculate_frequency_with_clustering(timestamps)
            max_creation_time_observed = calculate_max_creation_time(timestamps)

            results.append({
                "folder_path": dirpath,
                "file_prefix": prefix,
                "frequency": frequency,
                "fallback_time": fallback_time if fallback_time else "N/A",
                "average_interval_hours": round(avg_interval, 2) if avg_interval is not None else "N/A",
                "most_common_times": ", ".join(common_times) if common_times else "N/A",
                "max_creation_time_observed": max_creation_time_observed,
                "average_file_size_kb": round(avg_size_kb, 2) if isinstance(avg_size_kb, (int, float)) else "N/A",
                "last_seen": datetime.fromtimestamp(max(timestamps)).strftime("%Y-%m-%d %H:%M:%S") if timestamps else "N/A"
            })

        # Save results to CSV
        df = pd.DataFrame(results)
        # Backup before writing
        backup_file_if_exists(output_csv_path)
        df.to_csv(output_csv_path, index=False)
        logging.info(f"File patterns report saved to {output_csv_path}")

    except FileNotFoundError:
        logging.error(f"Output path or directory does not exist for {output_csv_path}", exc_info=True)
    except PermissionError:
        logging.error(f"Permission denied when trying to write to {output_csv_path}", exc_info=True)
    except Exception as e:
        logging.error(f"Error during file pattern analysis: {e}", exc_info=True)

# --- Run the program ---
if __name__ == "__main__":
    logging.info(f"\n{'='*20} Starting File Pattern Analysis ({datetime.now()}) {'='*20}")
    if not ROOT_FOLDERS:
         logging.error("No valid root folders specified in config file '[Monitoring] root_folders'. Exiting.")
    else:
        analyze_file_history(ROOT_FOLDERS, OUTPUT_FILE_PATH)
    logging.info(f"{'='*20} File Pattern Analysis Completed ({datetime.now()}) {'='*20}")