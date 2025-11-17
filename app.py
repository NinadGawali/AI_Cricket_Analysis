# import os
# import sqlite3
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_community.utilities import SQLDatabase
# from langchain_groq import ChatGroq
# from langchain_community.agent_toolkits import create_sql_agent
# from langchain.agents import AgentType

# # 1️⃣ Load API Key
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     st.error("❌ GROQ_API_KEY not found in .env file!")
#     st.stop()

# # 2️⃣ Connect to SQLite DB
# DB_PATH = "icc_cricket.db"
# if not os.path.exists(DB_PATH):
#     st.error(f"❌ Database file '{DB_PATH}' not found. Run createdb.py first.")
#     st.stop()

# db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# # 3️⃣ Initialize Llama 3 (ChatGroq)
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",  # Or "llama-3.3-8b" for faster inference
#     api_key=groq_api_key,
#     temperature=0
# )

# # 4️⃣ Create SQL Agent
# agent_executor = create_sql_agent(
#     llm=llm,
#     db=db,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=False
# )

# # --- STREAMLIT UI ---

# st.set_page_config(page_title="Cricket DB Chatbot", page_icon="🏏", layout="centered")

# # Custom CSS for White + Blue theme
# st.markdown(
#     """
#     <style>
#         body { background-color: #ffffff; }
#         .stApp { background-color: #ffffff; }
#         .main { color: #003366; }
#         div[data-testid="stChatMessage"] {
#             border-radius: 12px;
#             padding: 0.5rem 1rem;
#             margin-bottom: 0.5rem;
#             animation: fadeIn 0.3s ease-in-out;
#         }
#         div[data-testid="stChatMessage"]:nth-child(odd) {
#             background-color: #e6f0ff;
#         }
#         @keyframes fadeIn {
#             from {opacity: 0; transform: translateY(10px);}
#             to {opacity: 1; transform: translateY(0);}
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.title("🏏 Cricket Database Chatbot")
# st.write("Ask any question about the ICC Cricket dataset. Powered by **Llama 3 + LangChain + SQLite**.")

# # Store chat history in session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Chat input
# user_query = st.text_input("🔎 Enter your question:", "")

# if user_query:
#     with st.spinner("Thinking..."):
#         try:
#             response = agent_executor.run(user_query)
#             st.session_state.chat_history.append(("You", user_query))
#             st.session_state.chat_history.append(("Bot", response))
#         except Exception as e:
#             st.error(f"❌ Error: {e}")

# # Display chat history with animations
# for sender, msg in st.session_state.chat_history:
#     if sender == "You":
#         st.markdown(f"<div style='color:#003366'><b>🙋 {sender}:</b> {msg}</div>", unsafe_allow_html=True)
#     else:
#         st.markdown(f"<div style='color:#000000'><b>🤖 {sender}:</b> {msg}</div>", unsafe_allow_html=True)





import os
import sqlite3
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import sys

# LangChain / SQL utilities
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="🏏 Cricket Analytics Hub",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# Custom CSS - Clean Light & Dark Blue Theme
# --------------------------
st.markdown("""
    <style>
        /* Main Background - Clean Light */
        .stApp {
            background: linear-gradient(135deg, #F5F7FA 0%, #E8EAF6 100%);
        }
        
        /* Sidebar - Professional Dark Blue */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1565C0 0%, #0D47A1 100%);
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        [data-testid="stSidebar"] .stRadio > label {
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        /* Headers - Clean and Bold */
        h1 {
            color: #0D47A1 !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }
        
        h2, h3 {
            color: #1565C0 !important;
            font-weight: 700 !important;
        }
        
        h5 {
            color: #1976D2 !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem;
        }
        
        /* Metrics - Clean and Prominent */
        [data-testid="stMetricValue"] {
            color: #1565C0 !important;
            font-size: 2.5rem !important;
            font-weight: 800 !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #424242 !important;
            font-weight: 600 !important;
        }
        
        /* Cards/Containers - Clean White Cards */
        div[data-testid="stExpander"] {
            background-color: white;
            border-radius: 12px;
            border: 1px solid #BBDEFB;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin: 0.5rem 0;
        }
        
        /* Chat Messages - Clean Design */
        .chat-message {
            padding: 1.2rem;
            border-radius: 12px;
            margin: 0.8rem 0;
            animation: slideIn 0.3s ease-out;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .user-message {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            color: white;
            margin-left: 15%;
            border: none;
        }
        
        .bot-message {
            background: white;
            color: #212121;
            border: 2px solid #90CAF9;
            margin-right: 15%;
        }
        
        /* Code Blocks - Clean Syntax Highlighting */
        .stCodeBlock {
            background: #263238 !important;
            border-radius: 8px;
            border: 1px solid #37474F;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        
        code {
            color: #4FC3F7 !important;
            font-family: 'Fira Code', 'Courier New', monospace;
            font-size: 0.9rem;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(15px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Buttons - Modern and Clean */
        .stButton>button {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(21, 101, 192, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(21, 101, 192, 0.4);
            background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
        }
        
        /* Input Fields - Clean */
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 2px solid #BBDEFB;
            padding: 0.6rem;
            font-size: 1rem;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #1976D2;
            box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.1);
        }
        
        /* Tables - Clean Design */
        .dataframe {
            border: 2px solid #BBDEFB !important;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .dataframe thead tr th {
            background: #E3F2FD !important;
            color: #0D47A1 !important;
            font-weight: 700 !important;
            padding: 12px !important;
        }
        
        .dataframe tbody tr:hover {
            background: #F5F7FA !important;
        }
        
        /* Tabs - Clean Design */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: white;
            border-radius: 10px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            color: #1565C0;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
            color: white !important;
        }
        
        /* Remove extra spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Plotly charts background */
        .js-plotly-plot {
            background: white !important;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Load Environment
# --------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file!")
    st.stop()

# --------------------------
# Database Setup
# --------------------------
DB_PATH = "cricket_matches.db"
if not os.path.exists(DB_PATH):
    st.error(f"❌ Database file '{DB_PATH}' not found!")
    st.stop()

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# --------------------------
# Import Gemini LLM
# --------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0
    )
except Exception as e:
    st.error(f"❌ Failed to initialize Gemini LLM: {e}")
    st.stop()

# --------------------------
# Create SQL Agent with verbose output capture
# --------------------------
class OutputCapture:
    def __init__(self):
        self.output = []
    
    def write(self, text):
        self.output.append(text)
    
    def flush(self):
        pass
    
    def get_output(self):
        return ''.join(self.output)

try:
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
except Exception as e:
    st.error(f"❌ Failed to create SQL agent: {e}")
    st.stop()

# --------------------------
# Helper Functions
# --------------------------
def get_db_schema():
    """Get database schema information"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
    tables = cursor.fetchall()
    
    schema_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f'PRAGMA table_info({table_name})')
        columns = cursor.fetchall()
        schema_info[table_name] = [(col[1], col[2]) for col in columns]
    
    conn.close()
    return schema_info

def execute_query(query):
    """Execute SQL query and return results"""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return None

def run_agent_with_query_capture(user_query):
    """Run agent and capture the generated SQL query and raw results"""
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = OutputCapture()
    
    try:
        response = agent_executor.run(user_query)
        verbose_output = captured_output.get_output()
        sys.stdout = old_stdout
        
        # Extract SQL query from verbose output
        sql_query = None
        sql_result = None
        
        if "Action Input:" in verbose_output:
            lines = verbose_output.split('\n')
            for i, line in enumerate(lines):
                if "Action Input:" in line:
                    # Get the query from the next few lines until we hit "Observation:"
                    query_lines = []
                    for j in range(i+1, min(i+20, len(lines))):
                        if "Observation:" in lines[j]:
                            break
                        if lines[j].strip() and "Action:" not in lines[j]:
                            query_lines.append(lines[j].strip())
                    sql_query = ' '.join(query_lines).replace('"', '').replace("'", "").strip()
                    
                    # Extract the observation (SQL result)
                    for j in range(i+1, min(i+30, len(lines))):
                        if "Observation:" in lines[j]:
                            result_lines = []
                            for k in range(j+1, min(j+20, len(lines))):
                                if "Thought:" in lines[k] or "Action:" in lines[k]:
                                    break
                                if lines[k].strip():
                                    result_lines.append(lines[k].strip())
                            sql_result = '\n'.join(result_lines)
                            break
                    break
        
        return response, sql_query, sql_result, verbose_output
    except Exception as e:
        sys.stdout = old_stdout
        raise e

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("🏏 Cricket Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📊 Navigate",
    ["🏠 Home", "💬 AI Chat Assistant", "📈 Standard Analytics", "🗄️ Database Schema"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 About")
st.sidebar.info(
    "🤖 **Powered by Gemini 2.0 Flash Exp**\n\n"
    "💾 **SQLite Database**\n\n"
    "🔗 **LangChain SQL Agent**\n\n"
    "📊 **Plotly Visualizations**"
)

# --------------------------
# Initialize Session State
# --------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------
# PAGE 1: HOME
# --------------------------
if page == "🏠 Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🏏 Cricket Analytics Hub</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #1565C0;'>Powered by AI & Advanced Data Analytics</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    conn = sqlite3.connect(DB_PATH)
    total_matches = pd.read_sql_query("SELECT COUNT(*) as count FROM matches", conn).iloc[0]['count']
    total_players = pd.read_sql_query("SELECT COUNT(DISTINCT player_name) as count FROM players", conn).iloc[0]['count']
    total_venues = pd.read_sql_query("SELECT COUNT(DISTINCT venue) as count FROM matches", conn).iloc[0]['count']
    conn.close()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Matches", f"{total_matches:,}")
    with col2:
        st.metric("👥 Total Players", f"{total_players:,}")
    with col3:
        st.metric("🏟️ Venues", f"{total_venues:,}")
    with col4:
        st.metric("🤖 AI Model", "Gemini 2.0 Flash")
    
    st.markdown("---")
    
    # Feature Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💬 AI Chat Assistant")
        st.info(
            "🤖 Ask natural language questions about cricket data\n\n"
            "🔍 Get instant SQL-powered insights\n\n"
            "📊 View generated queries & results"
        )
    
    with col2:
        st.markdown("### 📈 Standard Analytics")
        st.success(
            "📊 Pre-built visualizations & insights\n\n"
            "🎯 Top performers & match statistics\n\n"
            "📉 Interactive charts & graphs"
        )
    
    st.markdown("---")
    
    # Quick Start
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **💬 AI Chat Assistant** - Ask questions like "Who won the most matches?" or "Show me top run scorers"
    2. **📈 Standard Analytics** - Explore pre-built visualizations and insights
    3. **🗄️ Database Schema** - View the complete database structure
    """)

# --------------------------
# PAGE 2: AI CHAT ASSISTANT
# --------------------------
elif page == "💬 AI Chat Assistant":
    st.title("💬 AI Cricket Assistant")
    st.markdown("Ask me anything about the cricket database! 🏏")
    
    # Chat input
    user_query = st.text_input("🔎 Your Question:", placeholder="e.g., Which team has won the most matches?", key="chat_input")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🚀 Ask", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and user_query:
        with st.spinner("🤔 Thinking..."):
            try:
                response, sql_query, sql_result, verbose_output = run_agent_with_query_capture(user_query)
                st.session_state.chat_history.append({
                    "user": user_query,
                    "bot": response,
                    "sql": sql_query,
                    "sql_result": sql_result,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Display chat history
    st.markdown("---")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        # User message
        st.markdown(f"""
            <div class="chat-message user-message">
                <b>🙋 You</b> <span style="float: right; font-size: 0.8rem; opacity: 0.8;">{chat['timestamp']}</span>
                <p style="margin-top: 0.5rem;">{chat['user']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Bot message - Parsed Answer
        st.markdown(f"""
            <div class="chat-message bot-message">
                <b>🤖 AI Assistant</b>
                <p style="margin-top: 0.5rem;">{chat['bot']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # SQL Query and Results Section
        if chat.get('sql') or chat.get('sql_result'):
            col1, col2 = st.columns(2)
            
            with col1:
                if chat.get('sql'):
                    st.markdown("##### � Generated SQL Query")
                    st.code(chat['sql'], language='sql')
            
            with col2:
                if chat.get('sql_result'):
                    st.markdown("##### 📊 Raw SQL Output")
                    st.code(chat['sql_result'], language='text')
        
        st.markdown("<br>", unsafe_allow_html=True)

# --------------------------
# PAGE 3: STANDARD ANALYTICS
# --------------------------
elif page == "📈 Standard Analytics":
    st.title("📈 Standard Cricket Analytics")
    st.markdown("Explore pre-built insights and visualizations 📊")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Match Stats", "👤 Player Stats", "🏟️ Venue Analysis", "⚡ Performance Trends"])
    
    with tab1:
        st.markdown("### 🏆 Match Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Wins by team
            query = """
            SELECT winner as Team, COUNT(*) as Wins 
            FROM outcome 
            WHERE winner IS NOT NULL 
            GROUP BY winner 
            ORDER BY Wins DESC 
            LIMIT 10
            """
            df_wins = pd.read_sql_query(query, conn)
            
            fig = px.bar(
                df_wins, 
                x='Wins', 
                y='Team',
                orientation='h',
                title='🏆 Top 10 Teams by Wins',
                color='Wins',
                color_continuous_scale=['#90CAF9', '#1565C0'],
                text='Wins'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#0D47A1'),
                showlegend=False
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Toss decision impact
            query = """
            SELECT toss_decision as Decision, COUNT(*) as Count 
            FROM toss 
            GROUP BY toss_decision
            """
            df_toss = pd.read_sql_query(query, conn)
            
            fig = px.pie(
                df_toss,
                values='Count',
                names='Decision',
                title='⚡ Toss Decision Distribution',
                color_discrete_sequence=['#1976D2', '#64B5F6']
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#0D47A1')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Matches by type
        query = """
        SELECT match_type as Type, COUNT(*) as Matches 
        FROM matches 
        GROUP BY match_type
        """
        df_type = pd.read_sql_query(query, conn)
        
        fig = px.bar(
            df_type,
            x='Type',
            y='Matches',
            title='🎮 Matches by Type',
            color='Matches',
            color_continuous_scale=['#E3F2FD', '#0D47A1'],
            text='Matches'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#0D47A1'),
            showlegend=False
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 👤 Player Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Player of match awards
            query = """
            SELECT player_name as Player, COUNT(*) as Awards 
            FROM player_of_match 
            GROUP BY player_name 
            ORDER BY Awards DESC 
            LIMIT 15
            """
            df_pom = pd.read_sql_query(query, conn)
            
            fig = px.bar(
                df_pom,
                x='Awards',
                y='Player',
                orientation='h',
                title='⭐ Top 15 Players of the Match',
                color='Awards',
                color_continuous_scale=['#BBDEFB', '#1565C0'],
                text='Awards'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#0D47A1'),
                showlegend=False,
                height=600
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top run scorers
            query = """
            SELECT striker as Player, SUM(runs_off_bat) as Runs 
            FROM ball_by_ball 
            GROUP BY striker 
            ORDER BY Runs DESC 
            LIMIT 15
            """
            df_runs = pd.read_sql_query(query, conn)
            
            fig = px.bar(
                df_runs,
                x='Runs',
                y='Player',
                orientation='h',
                title='🏏 Top 15 Run Scorers',
                color='Runs',
                color_continuous_scale=['#E3F2FD', '#0D47A1'],
                text='Runs'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#0D47A1'),
                showlegend=False,
                height=600
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🏟️ Venue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Matches per venue
            query = """
            SELECT venue as Venue, COUNT(*) as Matches 
            FROM matches 
            WHERE venue IS NOT NULL
            GROUP BY venue 
            ORDER BY Matches DESC 
            LIMIT 12
            """
            df_venue = pd.read_sql_query(query, conn)
            
            fig = px.bar(
                df_venue,
                x='Matches',
                y='Venue',
                orientation='h',
                title='🏟️ Top Venues by Matches',
                color='Matches',
                color_continuous_scale=['#90CAF9', '#1565C0'],
                text='Matches'
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#0D47A1'),
                showlegend=False,
                height=500
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Matches per city
            query = """
            SELECT city as City, COUNT(*) as Matches 
            FROM matches 
            WHERE city IS NOT NULL
            GROUP BY city 
            ORDER BY Matches DESC 
            LIMIT 12
            """
            df_city = pd.read_sql_query(query, conn)
            
            fig = px.pie(
                df_city,
                values='Matches',
                names='City',
                title='🌆 Matches by City (Top 12)',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#0D47A1'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### ⚡ Performance Trends")
        
        # Matches over seasons
        query = """
        SELECT season as Season, COUNT(*) as Matches 
        FROM matches 
        WHERE season IS NOT NULL
        GROUP BY season 
        ORDER BY season
        """
        df_season = pd.read_sql_query(query, conn)
        
        fig = px.line(
            df_season,
            x='Season',
            y='Matches',
            title='📅 Matches Over Seasons',
            markers=True,
            line_shape='spline'
        )
        fig.update_traces(
            line=dict(color='#1565C0', width=3),
            marker=dict(size=10, color='#1976D2')
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#0D47A1'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Win method distribution
            query = """
            SELECT 
                CASE 
                    WHEN won_by_runs > 0 THEN 'By Runs'
                    WHEN won_by_wickets > 0 THEN 'By Wickets'
                    ELSE 'Other'
                END as Method,
                COUNT(*) as Count
            FROM outcome
            WHERE winner IS NOT NULL
            GROUP BY Method
            """
            df_method = pd.read_sql_query(query, conn)
            
            fig = px.pie(
                df_method,
                values='Count',
                names='Method',
                title='🎯 Win Method Distribution',
                color_discrete_sequence=['#1565C0', '#42A5F5', '#90CAF9']
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#0D47A1')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Extras distribution
            query = """
            SELECT 
                SUM(wides) as Wides,
                SUM(noballs) as NoBalls,
                SUM(byes) as Byes,
                SUM(legbyes) as LegByes
            FROM ball_by_ball
            """
            df_extras = pd.read_sql_query(query, conn)
            df_extras_melted = pd.melt(df_extras, var_name='Type', value_name='Count')
            
            fig = px.bar(
                df_extras_melted,
                x='Type',
                y='Count',
                title='⚡ Extras Distribution',
                color='Type',
                color_discrete_sequence=['#1976D2', '#42A5F5', '#64B5F6', '#90CAF9']
            )
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#0D47A1'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    conn.close()

# --------------------------
# PAGE 4: DATABASE SCHEMA
# --------------------------
elif page == "🗄️ Database Schema":
    st.title("🗄️ Database Schema")
    st.markdown("Complete structure of the cricket database 📚")
    
    schema_info = get_db_schema()
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Tables", len(schema_info))
    with col2:
        total_columns = sum(len(cols) for cols in schema_info.values())
        st.metric("🔢 Total Columns", total_columns)
    with col3:
        st.metric("💾 Database", "SQLite")
    
    st.markdown("---")
    
    # Display each table schema
    for table_name, columns in schema_info.items():
        with st.expander(f"📋 **{table_name.upper()}** ({len(columns)} columns)", expanded=False):
            df_schema = pd.DataFrame(columns, columns=['Column Name', 'Data Type'])
            df_schema.index = range(1, len(df_schema) + 1)
            
            st.dataframe(
                df_schema,
                use_container_width=True,
                height=(len(columns) + 1) * 35 + 3
            )
            
            # Sample data
            conn = sqlite3.connect(DB_PATH)
            sample_data = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
            conn.close()
            
            st.markdown("**Sample Data:**")
            st.dataframe(sample_data, use_container_width=True)
    
    # ER Diagram representation
    st.markdown("---")
    st.markdown("### 🔗 Table Relationships")
    st.info("""
    **Key Relationships:**
    - 🔑 `matches.match_id` → Primary key for match information
    - 🔗 `teams.match_id` → Links teams to matches
    - 🔗 `players.match_id` → Links players to matches
    - 🔗 `outcome.match_id` → Match results
    - 🔗 `toss.match_id` → Toss information
    - 🔗 `ball_by_ball.match_id` → Ball-by-ball data
    - 🔗 `player_of_match.match_id` → Award winners
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #1565C0; font-weight: 600;'>🏏 Cricket Analytics Hub | Powered by Gemini 2.0 Flash Exp & LangChain 🤖</p>",
    unsafe_allow_html=True
)
