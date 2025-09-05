import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DefaultConfig:
    """ Bot Configuration """
    AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING', 'your_connection_string_here')
    AZURE_STORAGE_CONTAINER_NAME = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'your_container_name_here')
    CLU_API_KEY = os.getenv("CLU_API_KEY", "")
    CLU_ENDPOINT = os.getenv("CLU_ENDPOINT", "")
    CLU_PROJECT_NAME = os.getenv("CLU_PROJECT_NAME", "")
    CLU_DEPLOYMENT_NAME = os.getenv("CLU_DEPLOYMENT_NAME", "")
    PORT = 3978
    APP_ID = os.getenv("MicrosoftAppId", "")
    APP_PASSWORD = os.getenv("MicrosoftAppPassword", "")
    AZURE_SQL_CONNECTIONSTRING = "Driver={SQL Server};Server=tcp:your_server.database.windows.net,1433;Database=your_database;Uid=your_username;Pwd=your_password;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    DATE_FORMATS = [
        "%Y-%m-%d",  # 2023-01-01
        "%m/%d/%Y",  # 01/01/2023
        "%d/%m/%Y",  # 01/01/2023
        "%Y/%m/%d",  # 2023/01/01
        "%d-%m-%Y",  # 01-01-2023
        "%m-%d-%Y",  # 01-01-2023
    ]
    # Load messages, SQL queries, and report configuration from config.json
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)
    MESSAGES = config_data['MESSAGES']
    SQL_QUERIES = config_data['SQL_QUERIES']
    REPORT_CONFIG = config_data['REPORT_CONFIG']
    KEY_MAPPINGS = config_data['KEYMAP_CONFIG']
