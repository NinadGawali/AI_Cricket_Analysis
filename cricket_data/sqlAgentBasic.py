import os
from dotenv import load_dotenv
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

# 1️⃣ Load API Key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")

# 2️⃣ Connect to SQLite Database
DB_PATH = "icc_cricket.db"
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"❌ Database file '{DB_PATH}' not found. Run createdb.py first.")

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
print("✅ Connected to SQLite DB")

# 3️⃣ Initialize Llama 3 (via ChatGroq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # use "llama3-8b-8192" if you want cheaper/faster
    api_key=groq_api_key,
    temperature=0
)

# 4️⃣ Create SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # set to False to hide chain-of-thought logs
)

# 5️⃣ Interactive Query Loop
print("\n🤖 Cricket DB Chatbot Ready!")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    query = input("🔎 Enter your question: ").strip()
    if query.lower() in ("exit", "quit"):
        print("👋 Exiting. Goodbye!")
        break
    try:
        response = agent_executor.run(query)
        print(f"🤖 Answer: {response}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
