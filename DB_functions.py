import sqlite3

def clear_db(db_path: str = "keyscout.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM listings")
    conn.commit()
    conn.close()
