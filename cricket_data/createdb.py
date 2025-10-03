import os
import sqlite3
import pandas as pd

# Change if needed
BASE_FOLDER = "."
DB_NAME = "icc_cricket.db"

def create_database():
    # Connect / create SQLite database
    conn = sqlite3.connect(DB_NAME)
    print(f"✅ Created database: {DB_NAME}")

    for root, dirs, files in os.walk(BASE_FOLDER):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                
                # Use folder name + filename as table name for uniqueness
                folder = os.path.basename(root)
                if folder == ".":  # if file is at root
                    table_name = file.replace(".csv", "")
                else:
                    table_name = f"{folder}_{file.replace('.csv', '')}"

                # Clean table name (remove spaces)
                table_name = table_name.replace(" ", "_")

                print(f"📥 Loading {file_path} → Table: {table_name}")
                
                # Read CSV
                df = pd.read_csv(file_path)

                # Store in database
                df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.close()
    print("✅ All tables created successfully!")

if __name__ == "__main__":
    create_database()
