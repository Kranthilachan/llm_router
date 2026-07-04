import sqlite3
from pathlib import Path
from app.config import ATLAS_DB_PATH

DB_PATH = Path(__file__).parent.parent.parent / ATLAS_DB_PATH

def get_connection():
    """Returns a new SQLite connection."""
    # check_same_thread=False allows FastAPI to use this across threads if needed, 
    # but we should still be careful with transactions.
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
