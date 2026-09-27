import sqlite3
import json
import pandas as pd
from datetime import datetime

class DocumentDatabase:
    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates SQLite tables for audit history if they do not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                document_number TEXT,
                vendor_name TEXT,
                total_amount REAL,
                ocr_confidence REAL,
                extraction_method TEXT,
                human_verified INTEGER,
                raw_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def save_record(self, filename: str, doc_number: str, vendor: str, total: float, confidence: float, method: str, verified: bool, payload: dict):
        """Saves a processed document record into the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO processed_docs 
            (filename, document_number, vendor_name, total_amount, ocr_confidence, extraction_method, human_verified, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filename,
            doc_number or "N/A",
            vendor or "N/A",
            total or 0.0,
            confidence or 0.0,
            method or "Unknown",
            1 if verified else 0,
            json.dumps(payload),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()

    def fetch_all_records(self) -> pd.DataFrame:
        """Retrieves history records as a Pandas DataFrame for Streamlit rendering."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT id, filename, document_number, vendor_name, total_amount, ocr_confidence, extraction_method, human_verified, created_at FROM processed_docs ORDER BY id DESC", 
            conn
        )
        conn.close()
        return df
