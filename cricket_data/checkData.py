import sqlite3
import pandas as pd

conn = sqlite3.connect("icc_cricket.db")

tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables:", tables)

# Preview a table
df = pd.read_sql("SELECT * FROM Batting_ODI_data LIMIT 5;", conn)
print(df)

conn.close()
