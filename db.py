import sqlite3

def setup_database():
    conn = sqlite3.connect("lead_generation.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS service_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone_number TEXT NOT NULL,
        email TEXT NOT NULL,
        description TEXT
    )
    """)
    conn.commit()
    conn.close()

# Run this once to create the table
setup_database()

