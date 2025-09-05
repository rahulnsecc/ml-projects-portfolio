from database import get_conn
from config import DefaultConfig

class ProcessUserRequestDAO:
    def fetch_transaction_data(self, start_date, end_date, additional_info):
        query = self.build_query(additional_info)
        print(f"[fetch_transaction_data] Query: {query} with start_date: {start_date}, end_date: {end_date}")
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (start_date, end_date))
            rows = cursor.fetchall()
            return rows

    def build_query(self, additional_info):
        base_query = DefaultConfig.SQL_QUERIES['transaction_query']
        for key, value in additional_info.items():
            base_query += f" AND {key} = '{value}'"
        return base_query
