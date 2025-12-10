"""
Cricket Analytics Flask Application
Supports all cricket formats: ODI, Test, T20 International, IPL
"""

import os
import sys
import sqlite3
import json
import logging
import re
from functools import wraps
from datetime import datetime, timedelta
from contextlib import contextmanager
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI

# --------------------------
# Load Environment
# --------------------------
load_dotenv()

application = Flask(__name__)
app = application  # Elastic Beanstalk expects 'application'

# Enable verbose logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
app.logger.setLevel(logging.INFO)

# --------------------------
# Configuration
# --------------------------
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "all_formats_cricket.db"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app.logger.info(f"[INIT] DB_PATH: {DB_PATH}")
app.logger.info(f"[INIT] DB exists: {os.path.exists(DB_PATH)}")
app.logger.info(f"[INIT] API Key loaded: {bool(GEMINI_API_KEY)}")

if not os.path.exists(DB_PATH):
    app.logger.error(f"[INIT] Database not found at {DB_PATH}")

# --------------------------
# Global Variables
# --------------------------
llm = None
agent_executor = None
db = None

# --------------------------
# Helper Class: Capture stdout
# --------------------------
class OutputCapture:
    """Captures stdout to extract SQL queries from verbose agent output"""
    def __init__(self):
        self.output = []
    
    def write(self, text):
        self.output.append(text)
    
    def flush(self):
        pass
    
    def get_output(self):
        return ''.join(self.output)

# --------------------------
# Helper Function: Extract SQL from verbose output
# --------------------------
def extract_sql_from_verbose(verbose_output):
    """Extract SQL query and raw results from agent's verbose output."""
    sql_query = None
    sql_result = None
    
    if "Action Input:" in verbose_output:
        lines = verbose_output.split('\n')
        for i, line in enumerate(lines):
            if "Action Input:" in line:
                same_line_query = line.split("Action Input:")[-1].strip()
                if same_line_query and "sql_db_query" not in same_line_query.lower():
                    sql_query = same_line_query
                else:
                    query_lines = []
                    for j in range(i+1, min(i+20, len(lines))):
                        if "Observation:" in lines[j]:
                            break
                        if "Action:" in lines[j]:
                            break
                        line_content = lines[j].strip()
                        if line_content:
                            query_lines.append(line_content)
                    if query_lines:
                        sql_query = ' '.join(query_lines)
                
                if sql_query:
                    sql_query = sql_query.strip().strip('"').strip("'").strip()
                
                for j in range(i+1, min(i+40, len(lines))):
                    if "Observation:" in lines[j]:
                        same_line_result = lines[j].split("Observation:")[-1].strip()
                        if same_line_result:
                            sql_result = same_line_result
                        else:
                            result_lines = []
                            for k in range(j+1, min(j+30, len(lines))):
                                if "Thought:" in lines[k] or "Action:" in lines[k]:
                                    break
                                line_content = lines[k].strip()
                                if line_content:
                                    result_lines.append(line_content)
                            sql_result = '\n'.join(result_lines) if result_lines else None
                        break
                break
    
    return sql_query, sql_result

# --------------------------
# Initialize Agent
# --------------------------
def init_agent():
    """Initialize the SQL Agent with proper configuration"""
    global llm, agent_executor, db
    
    app.logger.info("[INIT] Starting agent initialization...")
    
    if not GEMINI_API_KEY:
        app.logger.error("[INIT] GEMINI_API_KEY not found in environment")
        return False
    
    if not os.path.exists(DB_PATH):
        app.logger.error(f"[INIT] Database file not found: {DB_PATH}")
        return False

    try:
        # Use sample_rows_in_table_info=0 and ignore_tables to avoid reflection issues
        # Also use include_tables to only include valid tables
        valid_table_list = list(VALID_TABLES) if 'VALID_TABLES' in dir() else None
        
        db = SQLDatabase.from_uri(
            f"sqlite:///{DB_PATH}",
            sample_rows_in_table_info=0,  # Don't sample rows to avoid reflection errors
            view_support=False
        )
        app.logger.info("[INIT] SQLDatabase initialized successfully")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GEMINI_API_KEY,
            temperature=0
        )
        app.logger.info("[INIT] ChatGoogleGenerativeAI initialized with gemini-2.5-flash")
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        app.logger.info("[INIT] SQLDatabaseToolkit created")
        
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            handle_parsing_errors=True
        )
        app.logger.info("[INIT] SQL Agent created successfully")
        
        return True
        
    except Exception as e:
        app.logger.error(f"[INIT] Failed to initialize agent: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return False

# Initialize on startup
init_agent()

# --------------------------
# Database Helper
# --------------------------
@contextmanager
def get_db_context():
    """Context manager for database connections - ensures proper cleanup"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        if conn:
            conn.close()

def get_db_connection():
    """Get a direct SQLite connection for non-agent queries (legacy support)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --------------------------
# Simple In-Memory Cache
# --------------------------
_cache = {}
_cache_timestamps = {}
CACHE_DURATION = timedelta(hours=1)

def cache_result(key, duration=CACHE_DURATION):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = datetime.now()
            if key in _cache and key in _cache_timestamps:
                if now - _cache_timestamps[key] < duration:
                    app.logger.info(f"[CACHE] Returning cached result for {key}")
                    return _cache[key]
            
            result = func(*args, **kwargs)
            _cache[key] = result
            _cache_timestamps[key] = now
            app.logger.info(f"[CACHE] Cached new result for {key}")
            return result
        return wrapper
    return decorator

def clear_cache(key=None):
    """Clear cache - all or specific key"""
    global _cache, _cache_timestamps
    if key:
        _cache.pop(key, None)
        _cache_timestamps.pop(key, None)
    else:
        _cache = {}
        _cache_timestamps = {}

# --------------------------
# Security: Input Validation
# --------------------------
# Valid table names in the database
VALID_TABLES = {
    'odi_matches', 'odi_ball_by_ball', 'odi_outcome', 'odi_toss', 'odi_players', 'odi_ball_fielders', 'odi_player_of_match',
    't20_matches', 't20_ball_by_ball', 't20_outcome', 't20_toss', 't20_players', 't20_ball_fielders', 't20_player_of_match',
    'test_matches', 'test_ball_by_ball', 'test_outcome', 'test_toss', 'test_players', 'test_ball_fielders', 'test_player_of_match',
    'ipl_matches', 'ipl_ball_by_ball', 'ipl_outcome', 'ipl_toss', 'ipl_players', 'ipl_ball_fielders', 'ipl_player_of_match'
}

# Pattern for valid column names (alphanumeric and underscore only)
VALID_COLUMN_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

def validate_table_name(table_name):
    """Validate table name against allowlist"""
    return table_name in VALID_TABLES

def validate_column_name(column_name):
    """Validate column name format"""
    return bool(VALID_COLUMN_PATTERN.match(column_name))

def get_valid_columns_for_table(table_name):
    """Get list of valid columns for a specific table"""
    if not validate_table_name(table_name):
        return []
    with get_db_context() as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]

def sanitize_like_param(value):
    """Sanitize a value for use in LIKE clause"""
    # Escape special SQL LIKE characters
    value = str(value).replace('%', '\\%').replace('_', '\\_')
    return f"%{value}%"

# --------------------------
# Player Name Helpers
# --------------------------
def fuzzy_player_match(player_name):
    """
    Generate SQL LIKE pattern for fuzzy player name matching.
    Handles variations like 'S Tendulkar' vs 'Sachin Tendulkar'
    """
    # Clean the player name
    name = player_name.strip()
    # Return a pattern that matches the name anywhere
    return f"%{name}%"

def get_player_variations(conn, player_name):
    """
    Find all matching player names from the database using LIKE matching.
    Returns the most likely match based on frequency.
    """
    pattern = fuzzy_player_match(player_name)
    
    query = """
        SELECT player_name, COUNT(*) as freq FROM (
            SELECT striker as player_name FROM odi_ball_by_ball WHERE striker LIKE ?
            UNION ALL SELECT striker as player_name FROM t20_ball_by_ball WHERE striker LIKE ?
            UNION ALL SELECT striker as player_name FROM test_ball_by_ball WHERE striker LIKE ?
            UNION ALL SELECT striker as player_name FROM ipl_ball_by_ball WHERE striker LIKE ?
            UNION ALL SELECT bowler as player_name FROM odi_ball_by_ball WHERE bowler LIKE ?
            UNION ALL SELECT bowler as player_name FROM t20_ball_by_ball WHERE bowler LIKE ?
            UNION ALL SELECT bowler as player_name FROM test_ball_by_ball WHERE bowler LIKE ?
            UNION ALL SELECT bowler as player_name FROM ipl_ball_by_ball WHERE bowler LIKE ?
        ) GROUP BY player_name ORDER BY freq DESC LIMIT 5
    """
    
    df = pd.read_sql_query(query, conn, params=[pattern]*8)
    return df['player_name'].tolist() if not df.empty else [player_name]

# --------------------------
# Routes: Pages
# --------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/schema')
def schema_page():
    return render_template('schema.html')

@app.route('/query-runner')
def query_runner_page():
    return render_template('query_runner.html')

@app.route('/match-predictor')
def match_predictor_page():
    return render_template('match_predictor.html')

@app.route('/player-analysis')
def player_analysis_page():
    return render_template('player_analysis.html')

@app.route('/health')
def health():
    """Health check endpoint for EB/Docker"""
    return jsonify({
        "status": "healthy",
        "agent_initialized": agent_executor is not None,
        "database_exists": os.path.exists(DB_PATH)
    }), 200

# --------------------------
# API: Chat (NLP to SQL Agent)
# --------------------------
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """NLP to SQL Agent endpoint."""
    data = request.json
    user_query = data.get('message', '').strip()
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    if not agent_executor:
        return jsonify({
            "error": "Agent not initialized. Check server logs for details.",
            "hint": "Verify GEMINI_API_KEY is set and database exists."
        }), 500

    app.logger.info(f"[CHAT] Query: {user_query}")
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = OutputCapture()
    
    try:
        response = agent_executor.run(user_query)
        verbose_output = captured_output.get_output()
        sys.stdout = old_stdout
        
        sql_query, sql_result = extract_sql_from_verbose(verbose_output)
        
        app.logger.info(f"[CHAT] Extracted SQL: {sql_query[:100] if sql_query else 'None'}...")
        app.logger.info(f"[CHAT] Response: {response[:100]}...")
        
        return jsonify({
            "success": True,
            "response": response,
            "sql_query": sql_query,
            "sql_result": sql_result
        })
        
    except Exception as e:
        sys.stdout = old_stdout
        app.logger.error(f"[CHAT] Error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# --------------------------
# API: Analytics Data (Multi-format) - WITH CACHING
# --------------------------
def _fetch_analytics_data():
    """Internal function to fetch analytics data - called by cached wrapper"""
    app.logger.info("Fetching analytics data from database...")
    data = {}
    
    with get_db_context() as conn:
        # 1. Wins by Team (All Formats Combined)
        df_wins = pd.read_sql_query("""
            SELECT winner as Team, COUNT(*) as Wins FROM (
                SELECT winner FROM odi_outcome WHERE winner IS NOT NULL
                UNION ALL SELECT winner FROM t20_outcome WHERE winner IS NOT NULL
                UNION ALL SELECT winner FROM test_outcome WHERE winner IS NOT NULL
                UNION ALL SELECT winner FROM ipl_outcome WHERE winner IS NOT NULL
            )
            GROUP BY winner 
            ORDER BY Wins DESC 
            LIMIT 10
        """, conn)
        data['wins_by_team'] = df_wins.to_dict(orient='records')
        
        # 2. Toss Decision (All Formats)
        df_toss = pd.read_sql_query("""
            SELECT toss_decision as Decision, COUNT(*) as Count FROM (
                SELECT toss_decision FROM odi_toss
                UNION ALL SELECT toss_decision FROM t20_toss
                UNION ALL SELECT toss_decision FROM test_toss
                UNION ALL SELECT toss_decision FROM ipl_toss
            )
            GROUP BY toss_decision
        """, conn)
        data['toss_decision'] = df_toss.to_dict(orient='records')
        
        # 3. Matches by Format
        df_type = pd.read_sql_query("""
            SELECT 'ODI' as Type, COUNT(*) as Matches FROM odi_matches
            UNION ALL SELECT 'T20', COUNT(*) FROM t20_matches
            UNION ALL SELECT 'Test', COUNT(*) FROM test_matches
            UNION ALL SELECT 'IPL', COUNT(*) FROM ipl_matches
        """, conn)
        data['matches_by_type'] = df_type.to_dict(orient='records')
        
        # 4. Top Run Scorers (All Formats)
        df_runs = pd.read_sql_query("""
            SELECT Player, SUM(Runs) as Runs, SUM(Matches) as Matches FROM (
                SELECT striker as Player, SUM(runs_off_bat) as Runs, COUNT(DISTINCT match_id) as Matches FROM odi_ball_by_ball GROUP BY striker
                UNION ALL 
                SELECT striker, SUM(runs_off_bat), COUNT(DISTINCT match_id) FROM t20_ball_by_ball GROUP BY striker
                UNION ALL 
                SELECT striker, SUM(runs_off_bat), COUNT(DISTINCT match_id) FROM test_ball_by_ball GROUP BY striker
                UNION ALL 
                SELECT striker, SUM(runs_off_bat), COUNT(DISTINCT match_id) FROM ipl_ball_by_ball GROUP BY striker
            )
            GROUP BY Player 
            ORDER BY Runs DESC 
            LIMIT 15
        """, conn)
        data['top_scorers'] = df_runs.to_dict(orient='records')
        
        # 5. Matches per Season (ODI + T20 Int)
        df_season = pd.read_sql_query("""
            SELECT season as Season, SUM(matches) as Matches FROM (
                SELECT season, COUNT(*) as matches FROM odi_matches WHERE season IS NOT NULL GROUP BY season
                UNION ALL SELECT season, COUNT(*) FROM t20_matches WHERE season IS NOT NULL GROUP BY season
            )
            GROUP BY season 
            ORDER BY season DESC
            LIMIT 20
        """, conn)
        data['matches_per_season'] = df_season.to_dict(orient='records')

        # 6. Win Method (All Formats)
        df_method = pd.read_sql_query("""
            SELECT Method, SUM(cnt) as Count FROM (
                SELECT CASE WHEN won_by_runs > 0 THEN 'By Runs' WHEN won_by_wickets > 0 THEN 'By Wickets' ELSE 'Other' END as Method, COUNT(*) as cnt FROM odi_outcome WHERE winner IS NOT NULL GROUP BY Method
                UNION ALL SELECT CASE WHEN won_by_runs > 0 THEN 'By Runs' WHEN won_by_wickets > 0 THEN 'By Wickets' ELSE 'Other' END, COUNT(*) FROM t20_outcome WHERE winner IS NOT NULL GROUP BY 1
                UNION ALL SELECT CASE WHEN won_by_runs > 0 THEN 'By Runs' WHEN won_by_wickets > 0 THEN 'By Wickets' ELSE 'Other' END, COUNT(*) FROM test_outcome WHERE winner IS NOT NULL GROUP BY 1
                UNION ALL SELECT CASE WHEN won_by_runs > 0 THEN 'By Runs' WHEN won_by_wickets > 0 THEN 'By Wickets' ELSE 'Other' END, COUNT(*) FROM ipl_outcome WHERE winner IS NOT NULL GROUP BY 1
            )
            GROUP BY Method
        """, conn)
        data['win_method'] = df_method.to_dict(orient='records')

        # 7. Top Wicket Takers (All Formats)
        df_wickets = pd.read_sql_query("""
            SELECT Player, SUM(Wickets) as Wickets, SUM(Matches) as Matches FROM (
                SELECT bowler as Player, COUNT(*) as Wickets, COUNT(DISTINCT match_id) as Matches 
                FROM odi_ball_by_ball WHERE wicket_type IS NOT NULL AND wicket_type != '' GROUP BY bowler
                UNION ALL 
                SELECT bowler, COUNT(*), COUNT(DISTINCT match_id) 
                FROM t20_ball_by_ball WHERE wicket_type IS NOT NULL AND wicket_type != '' GROUP BY bowler
                UNION ALL 
                SELECT bowler, COUNT(*), COUNT(DISTINCT match_id) 
                FROM test_ball_by_ball WHERE wicket_type IS NOT NULL AND wicket_type != '' GROUP BY bowler
                UNION ALL 
                SELECT bowler, COUNT(*), COUNT(DISTINCT match_id) 
                FROM ipl_ball_by_ball WHERE wicket_type IS NOT NULL AND wicket_type != '' GROUP BY bowler
            )
            GROUP BY Player 
            ORDER BY Wickets DESC 
            LIMIT 15
        """, conn)
        data['top_wicket_takers'] = df_wickets.to_dict(orient='records')

        # 8. Matches by City (All Formats)
        df_city = pd.read_sql_query("""
            SELECT city as City, COUNT(*) as Matches FROM (
                SELECT city FROM odi_matches WHERE city IS NOT NULL
                UNION ALL SELECT city FROM t20_matches WHERE city IS NOT NULL
                UNION ALL SELECT city FROM test_matches WHERE city IS NOT NULL
                UNION ALL SELECT city FROM ipl_matches WHERE city IS NOT NULL
            )
            GROUP BY city 
            ORDER BY Matches DESC 
            LIMIT 10
        """, conn)
        data['matches_by_city'] = df_city.to_dict(orient='records')

        # 9. Player of Match Awards (All Formats)
        df_pom = pd.read_sql_query("""
            SELECT player_name as Player, COUNT(*) as Awards FROM (
                SELECT player_name FROM ipl_player_of_match
                UNION ALL SELECT player_name FROM t20_player_of_match
                UNION ALL SELECT player_name FROM odi_player_of_match
                UNION ALL SELECT player_name FROM test_player_of_match
            )
            GROUP BY player_name 
            ORDER BY Awards DESC 
            LIMIT 10
        """, conn)
        data['player_of_match'] = df_pom.to_dict(orient='records')

        # 10. Format-wise Average Runs per Match
        df_avg = pd.read_sql_query("""
            SELECT 'T20 Int' as Format, ROUND(AVG(total_runs), 0) as AvgRuns FROM (SELECT match_id, SUM(runs_off_bat+extras) as total_runs FROM t20_ball_by_ball GROUP BY match_id)
            UNION ALL SELECT 'ODI', ROUND(AVG(total_runs), 0) FROM (SELECT match_id, SUM(runs_off_bat+extras) as total_runs FROM odi_ball_by_ball GROUP BY match_id)
            UNION ALL SELECT 'Test', ROUND(AVG(total_runs), 0) FROM (SELECT match_id, SUM(runs_off_bat+extras) as total_runs FROM test_ball_by_ball GROUP BY match_id)
            UNION ALL SELECT 'IPL', ROUND(AVG(total_runs), 0) FROM (SELECT match_id, SUM(runs_off_bat+extras) as total_runs FROM ipl_ball_by_ball GROUP BY match_id)
        """, conn)
        data['avg_runs_per_format'] = df_avg.to_dict(orient='records')

        # 11. Dismissal Types (IPL - Most Common)
        df_dismissal = pd.read_sql_query("""
            SELECT wicket_type as Type, COUNT(*) as Count 
            FROM ipl_ball_by_ball 
            WHERE wicket_type IS NOT NULL AND wicket_type != ''
            GROUP BY wicket_type 
            ORDER BY Count DESC 
            LIMIT 8
        """, conn)
        data['dismissal_types'] = df_dismissal.to_dict(orient='records')

        # 12. Top 10 Fielders (All Formats)
        df_fielders = pd.read_sql_query("""
            SELECT fielder_name as Fielder, COUNT(*) as Dismissals FROM (
                SELECT fielder_name FROM t20_ball_fielders
                UNION ALL SELECT fielder_name FROM odi_ball_fielders
                UNION ALL SELECT fielder_name FROM test_ball_fielders
                UNION ALL SELECT fielder_name FROM ipl_ball_fielders
            )
            GROUP BY fielder_name 
            ORDER BY Dismissals DESC 
            LIMIT 10
        """, conn)
        data['top_fielders'] = df_fielders.to_dict(orient='records')

        # 13. IPL Teams Performance (Batting First Win %)
        df_ipl_teams = pd.read_sql_query("""
            SELECT 
                team_name as Team, 
                COUNT(*) as Matches,
                SUM(CASE WHEN team_name = winner THEN 1 ELSE 0 END) as Wins,
                ROUND(100.0 * SUM(CASE WHEN team_name = winner THEN 1 ELSE 0 END) / COUNT(*), 1) as WinPct
            FROM ipl_teams t 
            JOIN ipl_outcome o ON o.match_id = t.match_id
            WHERE winner IS NOT NULL
            GROUP BY team_name 
            HAVING Matches >= 20
            ORDER BY WinPct DESC
        """, conn)
        data['ipl_team_performance'] = df_ipl_teams.to_dict(orient='records')

        # 14. Highest Totals per Format
        df_high_totals = pd.read_sql_query("""
            SELECT Format, batting_team as Team, match_id as MatchID, total_runs as TotalRuns FROM (
                SELECT 'T20 Int' as Format, match_id, batting_team, SUM(runs_off_bat+extras) as total_runs FROM t20_ball_by_ball GROUP BY match_id, batting_team ORDER BY total_runs DESC LIMIT 1
            )
            UNION ALL
            SELECT Format, batting_team, match_id, total_runs FROM (
                SELECT 'ODI' as Format, match_id, batting_team, SUM(runs_off_bat+extras) as total_runs FROM odi_ball_by_ball GROUP BY match_id, batting_team ORDER BY total_runs DESC LIMIT 1
            )
            UNION ALL
            SELECT Format, batting_team, match_id, total_runs FROM (
                SELECT 'IPL' as Format, match_id, batting_team, SUM(runs_off_bat+extras) as total_runs FROM ipl_ball_by_ball GROUP BY match_id, batting_team ORDER BY total_runs DESC LIMIT 1
            )
        """, conn)
        data['highest_totals'] = df_high_totals.to_dict(orient='records')

        # 15. Matches with Super Over (innings > 2)
        df_super_over = pd.read_sql_query("""
            SELECT Format, COUNT(*) as Matches FROM (
                SELECT 'T20 Int' as Format FROM (SELECT DISTINCT match_id FROM t20_ball_by_ball WHERE innings > 2)
                UNION ALL SELECT 'ODI' FROM (SELECT DISTINCT match_id FROM odi_ball_by_ball WHERE innings > 2)
                UNION ALL SELECT 'IPL' FROM (SELECT DISTINCT match_id FROM ipl_ball_by_ball WHERE innings > 2)
            )
            GROUP BY Format
        """, conn)
        data['super_over_matches'] = df_super_over.to_dict(orient='records')

    return data

@app.route('/api/analytics-data')
@cache_result('analytics_data', CACHE_DURATION)
def api_analytics_data():
    """Pre-built analytics queries for all cricket formats - CACHED for 1 hour"""
    try:
        data = _fetch_analytics_data()
        if isinstance(data, tuple):  # Error response
            return data
        return jsonify(data)
    except Exception as e:
        app.logger.error(f"Error in analytics data: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear-cache', methods=['POST'])
def api_clear_cache():
    """Admin endpoint to clear analytics cache"""
    clear_cache()
    return jsonify({"message": "Cache cleared successfully"})


# --------------------------
# API: Schema Info
# --------------------------
@app.route('/api/schema-info')
def api_schema_info():
    """Return database schema information"""
    with get_db_context() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name")
        tables = cursor.fetchall()
        
        schema = {}
        for table in tables:
            t_name = table['name']
            cursor.execute(f"PRAGMA table_info({t_name})")
            cols = cursor.fetchall()
            schema[t_name] = [{"name": c[1], "type": c[2]} for c in cols]
    
    return jsonify(schema)

# --------------------------
# API: Query Runner - SECURE VERSION
# --------------------------
@app.route('/api/query-runner/tables')
def api_query_tables():
    """Get list of available tables - returns only valid tables"""
    with get_db_context() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' ORDER BY name")
        tables = [row['name'] for row in cursor.fetchall()]
        # Filter to only return valid tables
        return jsonify([t for t in tables if t in VALID_TABLES])

@app.route('/api/query-runner/columns/<table_name>')
def api_query_columns(table_name):
    """Get columns for a table - validates table name first"""
    # SECURITY: Validate table name against allowlist
    if not validate_table_name(table_name):
        return jsonify({"error": "Invalid table name"}), 400
    
    with get_db_context() as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return jsonify(columns)

@app.route('/api/query-runner/execute', methods=['POST'])
def api_query_execute():
    """Execute a safe parameterized query with full input validation"""
    data = request.json
    table = data.get('table')
    columns = data.get('columns', [])
    filters = data.get('filters', [])
    limit = min(int(data.get('limit', 100)), 1000)  # Cap at 1000 rows
    
    if not table or not columns:
        return jsonify({"error": "Table and columns are required"}), 400
    
    # SECURITY: Validate table name against allowlist
    if not validate_table_name(table):
        return jsonify({"error": "Invalid table name"}), 400
    
    # SECURITY: Get valid columns for this table and validate requested columns
    valid_columns = get_valid_columns_for_table(table)
    if not valid_columns:
        return jsonify({"error": "Could not retrieve table schema"}), 500
    
    # Validate each requested column
    for col in columns:
        if col not in valid_columns:
            return jsonify({"error": f"Invalid column name: {col}"}), 400
    
    with get_db_context() as conn:
        try:
            # SECURITY: Columns are validated, safe to use in query
            cols_str = ", ".join(columns)
            query = f"SELECT {cols_str} FROM {table}"
            params = []
            
            if filters:
                where_clauses = []
                for f in filters:
                    col = f.get('column', '')
                    op = f.get('operator', '')
                    val = f.get('value', '')
                    
                    # SECURITY: Validate filter column against table's valid columns
                    if col not in valid_columns:
                        return jsonify({"error": f"Invalid filter column: {col}"}), 400
                    
                    # SECURITY: Validate operator against allowlist
                    if op not in ['=', '>', '<', '>=', '<=', 'LIKE', '!=']:
                        return jsonify({"error": f"Invalid operator: {op}"}), 400
                    
                    # Column is validated, safe to include in query
                    where_clauses.append(f"{col} {op} ?")
                    params.append(val)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn, params=params)
            return jsonify({
                "columns": columns,
                "rows": df.to_dict(orient='records'),
                "total": len(df)
            })
        except Exception as e:
            app.logger.error(f"Query execution error: {e}")
            return jsonify({"error": "Query execution failed"}), 500

# --------------------------
# API: Match Predictor (Multi-format) - SECURE VERSION
# --------------------------
@app.route('/api/match-predictor/predict', methods=['POST'])
def api_match_predictor():
    """Match prediction using LLM with database context - uses parameterized queries"""
    data = request.json
    team1 = data.get('team1', '').strip()
    team2 = data.get('team2', '').strip()
    venue = data.get('venue', '').strip()
    format_type = data.get('format', 'all')
    
    # Input validation
    if not team1 or not team2:
        return jsonify({"error": "Team names are required"}), 400
    
    if not llm:
        return jsonify({"error": "LLM not initialized"}), 500
    
    def generate():
        try:
            conn = get_db_connection()
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing head-to-head records...'})}\n\n"
            
            # SECURE: Use parameterized queries to prevent SQL injection
            team1_pattern = sanitize_like_param(team1)
            team2_pattern = sanitize_like_param(team2)
            venue_pattern = sanitize_like_param(venue) if venue else "%"
            
            # Get head-to-head stats across formats using parameterized queries
            h2h_queries = []
            h2h_params = []
            for fmt, tbl in [('ODI', 'odi'), ('T20', 't20'), ('Test', 'test'), ('IPL', 'ipl')]:
                h2h_queries.append(f"""
                    SELECT '{fmt}' as format, m.match_id, m.date, m.venue, o.winner
                    FROM {tbl}_matches m
                    JOIN {tbl}_outcome o ON m.match_id = o.match_id
                    WHERE (m.venue LIKE ? ESCAPE '\\' OR m.venue LIKE ? ESCAPE '\\'
                           OR o.winner LIKE ? ESCAPE '\\' OR o.winner LIKE ? ESCAPE '\\')
                """)
                h2h_params.extend([team1_pattern, team2_pattern, team1_pattern, team2_pattern])
            
            h2h_sql = " UNION ALL ".join(h2h_queries) + " ORDER BY date DESC LIMIT 15"
            h2h_stats = pd.read_sql_query(h2h_sql, conn, params=h2h_params)
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'Found {len(h2h_stats)} relevant matches'})}\n\n"
            
            # Get venue stats using parameterized queries
            venue_queries = []
            venue_params = []
            for fmt, tbl in [('ODI', 'odi'), ('T20', 't20'), ('Test', 'test'), ('IPL', 'ipl')]:
                venue_queries.append(f"""
                    SELECT '{fmt}' as format, o.winner, COUNT(*) as wins
                    FROM {tbl}_matches m JOIN {tbl}_outcome o ON m.match_id = o.match_id
                    WHERE m.venue LIKE ? ESCAPE '\\' AND o.winner IS NOT NULL
                    GROUP BY o.winner
                """)
                venue_params.append(venue_pattern)
            
            venue_sql = " UNION ALL ".join(venue_queries) + " ORDER BY wins DESC LIMIT 10"
            venue_stats = pd.read_sql_query(venue_sql, conn, params=venue_params)
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'Venue statistics collected'})}\n\n"
            
            conn.close()
            
            yield f"data: {json.dumps({'type': 'status', 'message': 'AI is generating prediction...'})}\n\n"
            
            prompt = f"""You are an expert cricket analyst. Predict the outcome of {team1} vs {team2} at {venue}.

Head-to-Head Records: {h2h_stats.to_string() if not h2h_stats.empty else 'No data'}
Venue Stats: {venue_stats.to_string() if not venue_stats.empty else 'No data'}

Provide:
1. Win probabilities for each team
2. Key factors affecting the prediction
3. Historical context
4. Final prediction with reasoning"""

            try:
                for chunk in llm.stream(prompt):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if content:
                        yield f"data: {json.dumps({'type': 'content', 'text': content})}\n\n"
            except:
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                yield f"data: {json.dumps({'type': 'content', 'text': content})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), headers={
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    })

# --------------------------
# API: Player Analysis (Multi-format)
# --------------------------
@app.route('/api/player-comparison')
def api_player_comparison():
    """Compare two players' stats across all formats using fuzzy name matching"""
    p1 = request.args.get('p1')
    p2 = request.args.get('p2')
    format_type = request.args.get('format', 'all')
    
    if not p1 or not p2:
        return jsonify({"error": "Missing player names"}), 400
        
    conn = get_db_connection()
    try:
        # Get best matching player names using LIKE
        p1_matches = get_player_variations(conn, p1)
        p2_matches = get_player_variations(conn, p2)
        p1_actual = p1_matches[0] if p1_matches else p1
        p2_actual = p2_matches[0] if p2_matches else p2
        
        # Build queries based on format
        formats = []
        if format_type == 'all':
            formats = [('odi', 'ODI'), ('t20', 'T20'), ('test', 'Test'), ('ipl', 'IPL')]
        else:
            format_map = {'odi': ('odi', 'ODI'), 't20': ('t20', 'T20'), 'test': ('test', 'Test'), 'ipl': ('ipl', 'IPL')}
            if format_type in format_map:
                formats = [format_map[format_type]]
        
        batting_queries = []
        for tbl, fmt_name in formats:
            batting_queries.append(f"""
                SELECT '{fmt_name}' as format, striker as player,
                       SUM(runs_off_bat) as runs,
                       COUNT(DISTINCT match_id) as matches,
                       COUNT(*) as balls,
                       ROUND(SUM(runs_off_bat) * 100.0 / NULLIF(COUNT(*), 0), 2) as strike_rate
                FROM {tbl}_ball_by_ball
                WHERE striker LIKE ? OR striker LIKE ?
                GROUP BY striker
            """)
        
        batting_sql = " UNION ALL ".join(batting_queries)
        batting_params = []
        for _ in formats:
            batting_params.extend([fuzzy_player_match(p1_actual), fuzzy_player_match(p2_actual)])
        
        batting_df = pd.read_sql_query(batting_sql, conn, params=batting_params)
        
        bowling_queries = []
        for tbl, fmt_name in formats:
            bowling_queries.append(f"""
                SELECT '{fmt_name}' as format, bowler as player,
                       COUNT(*) as balls,
                       SUM(runs_off_bat) as runs_conceded,
                       COUNT(CASE WHEN wicket_type IS NOT NULL AND wicket_type != '' THEN 1 END) as wickets,
                       ROUND(SUM(runs_off_bat) * 6.0 / NULLIF(COUNT(*), 0), 2) as economy
                FROM {tbl}_ball_by_ball
                WHERE bowler LIKE ? OR bowler LIKE ?
                GROUP BY bowler
            """)
        
        bowling_sql = " UNION ALL ".join(bowling_queries)
        bowling_params = []
        for _ in formats:
            bowling_params.extend([fuzzy_player_match(p1_actual), fuzzy_player_match(p2_actual)])
        
        bowling_df = pd.read_sql_query(bowling_sql, conn, params=bowling_params)
        
        return jsonify({
            "batting": batting_df.to_dict(orient='records'),
            "bowling": bowling_df.to_dict(orient='records'),
            "matched_players": {"player1": p1_actual, "player2": p2_actual}
        })
    except Exception as e:
        app.logger.error(f"Error in player comparison: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/batter-vs-bowler')
def api_batter_vs_bowler():
    """Head-to-head stats between batter and bowler across all formats using fuzzy matching"""
    batter = request.args.get('batter')
    bowler = request.args.get('bowler')
    format_type = request.args.get('format', 'all')
    
    if not batter or not bowler:
        return jsonify({"error": "Missing batter or bowler name"}), 400
        
    conn = get_db_connection()
    try:
        # Get best matching player names
        batter_matches = get_player_variations(conn, batter)
        bowler_matches = get_player_variations(conn, bowler)
        batter_actual = batter_matches[0] if batter_matches else batter
        bowler_actual = bowler_matches[0] if bowler_matches else bowler
        
        formats = []
        if format_type == 'all':
            formats = [('odi', 'ODI'), ('t20', 'T20'), ('test', 'Test'), ('ipl', 'IPL')]
        else:
            format_map = {'odi': ('odi', 'ODI'), 't20': ('t20', 'T20'), 'test': ('test', 'Test'), 'ipl': ('ipl', 'IPL')}
            if format_type in format_map:
                formats = [format_map[format_type]]
        
        queries = []
        for tbl, fmt_name in formats:
            queries.append(f"""
                SELECT '{fmt_name}' as format,
                    COUNT(*) as balls_faced,
                    SUM(runs_off_bat) as runs_scored,
                    COUNT(CASE WHEN wicket_type IS NOT NULL AND wicket_type != '' AND player_dismissed LIKE ? THEN 1 END) as dismissals,
                    SUM(CASE WHEN runs_off_bat = 4 THEN 1 ELSE 0 END) as fours,
                    SUM(CASE WHEN runs_off_bat = 6 THEN 1 ELSE 0 END) as sixes
                FROM {tbl}_ball_by_ball
                WHERE striker LIKE ? AND bowler LIKE ?
            """)
        
        params = []
        batter_pattern = fuzzy_player_match(batter_actual)
        bowler_pattern = fuzzy_player_match(bowler_actual)
        for _ in formats:
            params.extend([batter_pattern, batter_pattern, bowler_pattern])
        
        df = pd.read_sql_query(" UNION ALL ".join(queries), conn, params=params)
        
        # Aggregate totals
        totals = {
            'balls_faced': int(df['balls_faced'].sum()),
            'runs_scored': int(df['runs_scored'].sum()),
            'dismissals': int(df['dismissals'].sum()),
            'fours': int(df['fours'].sum()),
            'sixes': int(df['sixes'].sum()),
            'by_format': df.to_dict(orient='records'),
            'matched_players': {'batter': batter_actual, 'bowler': bowler_actual}
        }
        
        return jsonify(totals)
    except Exception as e:
        app.logger.error(f"Error in batter vs bowler: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/player-venue')
def api_player_venue():
    """Player performance by venue across all formats using fuzzy matching"""
    player = request.args.get('player')
    format_type = request.args.get('format', 'all')
    
    if not player:
        return jsonify({"error": "Missing player name"}), 400
        
    conn = get_db_connection()
    try:
        # Get best matching player name
        player_matches = get_player_variations(conn, player)
        player_actual = player_matches[0] if player_matches else player
        
        formats = []
        if format_type == 'all':
            formats = [('odi', 'ODI'), ('t20', 'T20'), ('test', 'Test'), ('ipl', 'IPL')]
        else:
            format_map = {'odi': ('odi', 'ODI'), 't20': ('t20', 'T20'), 'test': ('test', 'Test'), 'ipl': ('ipl', 'IPL')}
            if format_type in format_map:
                formats = [format_map[format_type]]
        
        queries = []
        for tbl, fmt_name in formats:
            queries.append(f"""
                SELECT '{fmt_name}' as format, venue,
                    SUM(runs_off_bat) as runs,
                    COUNT(*) as balls,
                    ROUND(SUM(runs_off_bat) * 100.0 / NULLIF(COUNT(*), 0), 2) as strike_rate,
                    COUNT(DISTINCT match_id) as matches
                FROM {tbl}_ball_by_ball
                WHERE striker LIKE ?
                GROUP BY venue
            """)
        
        player_pattern = fuzzy_player_match(player_actual)
        params = [player_pattern] * len(formats)
        
        df = pd.read_sql_query(" UNION ALL ".join(queries), conn, params=params)
        
        # Aggregate by venue
        venue_stats = df.groupby('venue').agg({
            'runs': 'sum',
            'balls': 'sum',
            'matches': 'sum'
        }).reset_index()
        venue_stats['strike_rate'] = round(venue_stats['runs'] * 100.0 / venue_stats['balls'].replace(0, 1), 2)
        venue_stats = venue_stats.sort_values('runs', ascending=False).head(15)
        
        return jsonify(venue_stats.to_dict(orient='records'))
    except Exception as e:
        app.logger.error(f"Error in player venue: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/player-spin-pace')
def api_player_spin_pace():
    """Analyze player performance vs spin/pace using LLM classification"""
    player = request.args.get('player')
    format_type = request.args.get('format', 'all')
    
    if not player:
        return jsonify({"error": "Missing player name"}), 400
        
    conn = get_db_connection()
    try:
        # Get best matching player name
        player_matches = get_player_variations(conn, player)
        player_actual = player_matches[0] if player_matches else player
        
        formats = []
        if format_type == 'all':
            formats = [('odi', 'ODI'), ('t20', 'T20'), ('test', 'Test'), ('ipl', 'IPL')]
        else:
            format_map = {'odi': ('odi', 'ODI'), 't20': ('t20', 'T20'), 'test': ('test', 'Test'), 'ipl': ('ipl', 'IPL')}
            if format_type in format_map:
                formats = [format_map[format_type]]
        
        # Get bowlers faced using LIKE
        queries = []
        for tbl, _ in formats:
            queries.append(f"SELECT bowler, COUNT(*) as balls FROM {tbl}_ball_by_ball WHERE striker LIKE ? GROUP BY bowler")
        
        player_pattern = fuzzy_player_match(player_actual)
        params = [player_pattern] * len(formats)
        bowlers_df = pd.read_sql_query(" UNION ALL ".join(queries), conn, params=params)
        bowlers_df = bowlers_df.groupby('bowler').agg({'balls': 'sum'}).reset_index()
        bowlers_df = bowlers_df.sort_values('balls', ascending=False).head(25)
        bowlers_list = bowlers_df['bowler'].tolist()
        
        if not bowlers_list:
            return jsonify({"error": "No data found for player"}), 404

        if not llm:
            return jsonify({"error": "LLM not initialized"}), 500
             
        # Classify bowlers
        prompt = f"""Classify these cricket bowlers as 'Spin' or 'Pace'. Return ONLY a JSON object with bowler names as keys.
Bowlers: {', '.join(bowlers_list)}"""
        
        response = llm.invoke(prompt)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        classification = json.loads(content.strip())
        
        # Aggregate stats using LIKE
        stats_queries = []
        for tbl, _ in formats:
            placeholders = ','.join(['?'] * len(bowlers_list))
            stats_queries.append(f"""
                SELECT bowler, SUM(runs_off_bat) as runs, COUNT(*) as balls
                FROM {tbl}_ball_by_ball
                WHERE striker LIKE ? AND bowler IN ({placeholders})
                GROUP BY bowler
            """)
        
        stats_params = []
        for _ in formats:
            stats_params.append(player_pattern)
            stats_params.extend(bowlers_list)
        
        stats_df = pd.read_sql_query(" UNION ALL ".join(stats_queries), conn, params=stats_params)
        stats_df = stats_df.groupby('bowler').agg({'runs': 'sum', 'balls': 'sum'}).reset_index()
        
        spin_runs, spin_balls, pace_runs, pace_balls = 0, 0, 0, 0
        for _, row in stats_df.iterrows():
            b_type = classification.get(row['bowler'], 'Unknown')
            if b_type == 'Spin':
                spin_runs += row['runs']
                spin_balls += row['balls']
            elif b_type == 'Pace':
                pace_runs += row['runs']
                pace_balls += row['balls']
                
        return jsonify({
            "spin": {"runs": int(spin_runs), "balls": int(spin_balls), 
                     "avg_sr": round(spin_runs*100/spin_balls, 2) if spin_balls else 0},
            "pace": {"runs": int(pace_runs), "balls": int(pace_balls), 
                     "avg_sr": round(pace_runs*100/pace_balls, 2) if pace_balls else 0},
            "details": classification
        })
        
    except Exception as e:
        app.logger.error(f"Error in spin/pace analysis: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --------------------------
# API: Format-specific Stats
# --------------------------
@app.route('/api/format-stats/<format_type>')
def api_format_stats(format_type):
    """Get statistics for a specific format"""
    valid_formats = {'odi': 'odi', 't20': 't20', 'test': 'test', 'ipl': 'ipl'}
    if format_type not in valid_formats:
        return jsonify({"error": "Invalid format. Use: odi, t20, test, ipl"}), 400
    
    tbl = valid_formats[format_type]
    conn = get_db_connection()
    
    try:
        data = {}
        
        # Top scorers
        df_scorers = pd.read_sql_query(f"""
            SELECT striker as Player, SUM(runs_off_bat) as Runs, COUNT(DISTINCT match_id) as Matches
            FROM {tbl}_ball_by_ball
            GROUP BY striker ORDER BY Runs DESC LIMIT 10
        """, conn)
        data['top_scorers'] = df_scorers.to_dict(orient='records')
        
        # Top wicket takers
        df_wickets = pd.read_sql_query(f"""
            SELECT bowler as Player, COUNT(*) as Wickets
            FROM {tbl}_ball_by_ball
            WHERE wicket_type IS NOT NULL AND wicket_type != ''
            GROUP BY bowler ORDER BY Wickets DESC LIMIT 10
        """, conn)
        data['top_wicket_takers'] = df_wickets.to_dict(orient='records')
        
        # Wins by team
        df_wins = pd.read_sql_query(f"""
            SELECT winner as Team, COUNT(*) as Wins
            FROM {tbl}_outcome
            WHERE winner IS NOT NULL
            GROUP BY winner ORDER BY Wins DESC LIMIT 10
        """, conn)
        data['wins_by_team'] = df_wins.to_dict(orient='records')
        
        # Top venues
        df_venues = pd.read_sql_query(f"""
            SELECT venue as Venue, COUNT(*) as Matches
            FROM {tbl}_matches
            WHERE venue IS NOT NULL
            GROUP BY venue ORDER BY Matches DESC LIMIT 10
        """, conn)
        data['top_venues'] = df_venues.to_dict(orient='records')
        
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --------------------------
# API: Players List (for autocomplete)
# --------------------------
@app.route('/api/players')
def api_players():
    """Get list of players for autocomplete"""
    search = request.args.get('search', '')
    format_type = request.args.get('format', 'all')
    
    conn = get_db_connection()
    try:
        if format_type == 'all':
            query = """
                SELECT DISTINCT player_name FROM (
                    SELECT player_name FROM odi_players
                    UNION SELECT player_name FROM t20_players
                    UNION SELECT player_name FROM test_players
                    UNION SELECT player_name FROM ipl_players
                )
                WHERE player_name LIKE ?
                ORDER BY player_name
                LIMIT 50
            """
        else:
            format_map = {'odi': 'odi_players', 't20': 't20_players', 'test': 'test_players', 'ipl': 'ipl_players'}
            tbl = format_map.get(format_type, 'odi_players')
            query = f"""
                SELECT DISTINCT player_name 
                FROM {tbl}
                WHERE player_name LIKE ?
                ORDER BY player_name
                LIMIT 50
            """
        
        df = pd.read_sql_query(query, conn, params=[f'%{search}%'])
        return jsonify(df['player_name'].tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --------------------------
# Run Application
# --------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
