import sqlite3

conn = sqlite3.connect('all_formats_cricket.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cursor.fetchall()]
print("Tables in database:")
for t in tables:
    print(f"  - {t}")

print("\n" + "="*50)

# Get schema for each table
for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    print(f"\nTable: {table}")
    print(f"  Columns: {[c[1] for c in columns]}")
    
    # Get sample row count
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  Row count: {count}")

conn.close()
