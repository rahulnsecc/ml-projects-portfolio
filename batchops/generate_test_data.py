import os
import time
from datetime import datetime, timedelta

def create_folder_and_files(base_path, num_applications, num_days, before_batch_run=True):
    """
    Create folders and files with timestamps spanning the last `num_days` days for `num_applications` applications.

    Args:
        base_path (str): The base directory where folders and files will be created.
        num_applications (int): The number of applications for which folders and files are created.
        num_days (int): The number of days over which files are distributed.
        before_batch_run (bool): If True, create files for "before the batch run", otherwise for "after the batch run".
    """
    scenario = "before_batch_run" if before_batch_run else "after_batch_run"
    scenario_path = os.path.join(base_path, scenario)

    for app_num in range(1, num_applications + 1):
        app_name = f"app{app_num}"
        # Define folders for this application
        app_folders = {
            "input": os.path.join(scenario_path, "input", app_name),
            "output": os.path.join(scenario_path, "output", app_name),
            "backup": os.path.join(scenario_path, "outputBackup", app_name),
            "logs": os.path.join(scenario_path, "logs")
        }

        # Create folders
        for folder_name, folder_path in app_folders.items():
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created folder: {folder_path}")

        # Create files for the last `num_days` days
        for day in range(num_days):
            file_date = datetime.now() - timedelta(days=day)
            
            # Create daily files
            daily_file_name = f"{app_name}_daily_{file_date.strftime('%Y%m%d')}.txt"
            daily_file_path = os.path.join(app_folders["output"], daily_file_name)
            with open(daily_file_path, "w") as f:
                f.write(f"Daily file content for {app_name} on {file_date.strftime('%Y-%m-%d')}")
            timestamp = time.mktime(file_date.timetuple())
            os.utime(daily_file_path, (timestamp, timestamp))
            print(f"Created daily file: {daily_file_path} with timestamp: {file_date}")

            # Create weekly files (every 7th day)
            if day % 7 == 0:
                weekly_file_name = f"{app_name}_weekly_{file_date.strftime('%Y%m%d')}.txt"
                weekly_file_path = os.path.join(app_folders["output"], weekly_file_name)
                with open(weekly_file_path, "w") as f:
                    f.write(f"Weekly file content for {app_name} on {file_date.strftime('%Y-%m-%d')}")
                os.utime(weekly_file_path, (timestamp, timestamp))
                print(f"Created weekly file: {weekly_file_path} with timestamp: {file_date}")

            # Create monthly files (first day of the month)
            if file_date.day == 1:
                monthly_file_name = f"{app_name}_monthly_{file_date.strftime('%Y%m%d')}.txt"
                monthly_file_path = os.path.join(app_folders["output"], monthly_file_name)
                with open(monthly_file_path, "w") as f:
                    f.write(f"Monthly file content for {app_name} on {file_date.strftime('%Y-%m-%d')}")
                os.utime(monthly_file_path, (timestamp, timestamp))
                print(f"Created monthly file: {monthly_file_path} with timestamp: {file_date}")

        # Create a log file for the application
        log_file_name = f"{app_name}.log"
        log_file_path = os.path.join(app_folders["logs"], log_file_name)
        with open(log_file_path, "w") as f:
            f.write(f"Log content for {app_name}")
        os.utime(log_file_path, (timestamp, timestamp))
        print(f"Created log file: {log_file_path}")

if __name__ == "__main__":
    base_path = os.path.abspath("test_environment")
    num_applications = 10
    num_days = 15

    # Create setup for "before the batch run"
    create_folder_and_files(base_path, num_applications, num_days, before_batch_run=True)

    # Create setup for "after the batch run"
    create_folder_and_files(base_path, num_applications, num_days, before_batch_run=False)
