import sqlite3
import pandas as pd
import os

DB_PATH = "cricket_matches.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def test_analytics():
    conn = get_db_connection()
    try:
        print("Testing Wins by Team...")
        df_wins = pd.read_sql_query("SELECT winner as Team, COUNT(*) as Wins FROM outcome WHERE winner IS NOT NULL GROUP BY winner ORDER BY Wins DESC LIMIT 10", conn)
        print(df_wins)
        
        print("\nTesting Toss Decision...")
        df_toss = pd.read_sql_query("SELECT toss_decision as Decision, COUNT(*) as Count FROM toss GROUP BY toss_decision", conn)
        print(df_toss)

        print("\nTesting Matches by Type...")
        df_type = pd.read_sql_query("SELECT match_type as Type, COUNT(*) as Matches FROM matches GROUP BY match_type", conn)
        print(df_type)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_analytics()
