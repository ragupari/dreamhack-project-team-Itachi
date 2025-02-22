import sqlite3
conn = sqlite3.connect('dreamhack.db')
cur = conn.cursor()

# Example SQL to create a table
cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        username TEXT,
        password TEXT
    )
''')

cur.execute('''
    INSERT INTO users (name, age)
    VALUES (?, ?)
''', ('Parishith', 23, 'pari','pari'))

conn.commit()
