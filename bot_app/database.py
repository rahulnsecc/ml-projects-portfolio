import pyodbc
from config import DefaultConfig

CONFIG = DefaultConfig()

def get_conn():
    print("[get_conn] Establishing database connection")
    return pyodbc.connect(CONFIG.AZURE_SQL_CONNECTIONSTRING)
