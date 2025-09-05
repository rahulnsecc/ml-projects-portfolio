from database import get_conn
from config import DefaultConfig

class ProcessUserRequestDAO:
    def fetch_item_details(self, item: int, store: str):
        query = DefaultConfig.SQL_QUERIES['fetch_item_details']
        print(f"[fetch_item_details] Query: {query} with item: {item}, store: {store}")
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (item, store))
            row = cursor.fetchone()
            if row is None:
                return '0'
            return row.CATEGORY

    def fetch_item_inventory(self, item, store, category):
        query = DefaultConfig.SQL_QUERIES['fetch_item_inventory']
        print(f"[fetch_item_inventory] Query: {query} with item: {item}, store: {store}, category: {category}")
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (item, store, category))
            row = cursor.fetchone()
            if row:
                return row.Inventory, row.WeeklySales
            return '0', '0'
