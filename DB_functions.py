import sqlite3

def clear_db(db_path: str = "keyscout.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM listings")
    cursor.execute("DELETE FROM listings_enriched")
    # cursor.execute(f"DROP TABLE IF EXISTS listings")
    # cursor.execute(f"DROP TABLE IF EXISTS listings_enriched")
    conn.commit()
    conn.close()


