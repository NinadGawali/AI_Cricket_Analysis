import os
import sqlite3
import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

# 1️⃣ Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env file!")
    st.stop()

# 2️⃣ Connect to SQLite DB
DB_PATH = "icc_cricket.db"
if not os.path.exists(DB_PATH):
    st.error(f"❌ Database file '{DB_PATH}' not found. Run createdb.py first.")
    st.stop()

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# 3️⃣ Initialize Llama 3 (ChatGroq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Or "llama-3.3-8b" for faster inference
    api_key=groq_api_key,
    temperature=0
)

# 4️⃣ Create SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# --- STREAMLIT UI ---

st.set_page_config(page_title="Cricket DB Chatbot", page_icon="🏏", layout="centered")

# Custom CSS for White + Blue theme
st.markdown(
    """
    <style>
        body { background-color: #ffffff; }
        .stApp { background-color: #ffffff; }
        .main { color: #003366; }
        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 0.5rem 1rem;
            margin-bottom: 0.5rem;
            animation: fadeIn 0.3s ease-in-out;
        }
        div[data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #e6f0ff;
        }
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏏 Cricket Database Chatbot")
st.write("Ask any question about the ICC Cricket dataset. Powered by **Llama 3 + LangChain + SQLite**.")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_query = st.text_input("🔎 Enter your question:", "")

if user_query:
    with st.spinner("Thinking..."):
        try:
            response = agent_executor.run(user_query)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("Bot", response))
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Display chat history with animations
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"<div style='color:#003366'><b>🙋 {sender}:</b> {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:#000000'><b>🤖 {sender}:</b> {msg}</div>", unsafe_allow_html=True)




