# view_db.py - Edupredict AI
# Inspect database tables (users, students, academic_records, prediction_results, hod_requests).

import sqlite3
import pandas as pd
from database import DB_PATH

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    print("Edupredict AI – Tables:", [t[0] for t in tables])
    for (tname,) in tables:
        df = pd.read_sql(f"SELECT * FROM {tname}", conn)
        print(f"\n📌 {tname}:\n", df.head(20))
    conn.close()
