"""
Cricket Analytics Flask Application
Fixed version with proper SQL Agent implementation (inspired by Streamlit app.py)
"""

import os
import sys
import sqlite3
import json
import logging
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
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "cricket_matches.db"))
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
# Helper Class: Capture stdout (like Streamlit app)
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
    """
    Extract SQL query and raw results from agent's verbose output.
    This matches the logic in the Streamlit app's run_agent_with_query_capture function.
    """
    sql_query = None
    sql_result = None
    
    if "Action Input:" in verbose_output:
        lines = verbose_output.split('\n')
        for i, line in enumerate(lines):
            if "Action Input:" in line:
                # Get the query - it might be on the same line or next lines
                # First check if it's on the same line after "Action Input:"
                same_line_query = line.split("Action Input:")[-1].strip()
                if same_line_query and "sql_db_query" not in same_line_query.lower():
                    sql_query = same_line_query
                else:
                    # Get query from next few lines until we hit "Observation:"
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
                
                # Clean up the SQL query
                if sql_query:
                    sql_query = sql_query.strip().strip('"').strip("'").strip()
                
                # Extract the observation (SQL result)
                for j in range(i+1, min(i+40, len(lines))):
                    if "Observation:" in lines[j]:
                        # Result might be on the same line
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
# Initialize Agent (like Streamlit app)
# --------------------------
def init_agent():
    """Initialize the SQL Agent with proper configuration matching Streamlit app"""
    global llm, agent_executor, db
    
    app.logger.info("[INIT] Starting agent initialization...")
    
    if not GEMINI_API_KEY:
        app.logger.error("[INIT] GEMINI_API_KEY not found in environment")
        return False
    
    if not os.path.exists(DB_PATH):
        app.logger.error(f"[INIT] Database file not found: {DB_PATH}")
        return False

    try:
        # Initialize database connection (same as Streamlit)
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        app.logger.info("[INIT] SQLDatabase initialized successfully")
        
        # Initialize LLM (matching Streamlit configuration)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Same model as Streamlit
            api_key=GEMINI_API_KEY,
            temperature=0  # Same temperature as Streamlit for consistent results
        )
        app.logger.info("[INIT] ChatGoogleGenerativeAI initialized with gemini-2.0-flash")
        
        # Create SQL Toolkit (required for newer LangChain versions)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        app.logger.info("[INIT] SQLDatabaseToolkit created")
        
        # Create SQL Agent with toolkit
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,  # Required for capturing SQL queries
            handle_parsing_errors=True  # Prevents hallucination errors
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
def get_db_connection():
    """Get a direct SQLite connection for non-agent queries"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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
# API: Chat (NLP to SQL Agent) - Fixed Implementation
# --------------------------
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    NLP to SQL Agent endpoint.
    Implements the same logic as Streamlit's run_agent_with_query_capture function.
    """
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
    
    # Capture stdout to extract SQL (same as Streamlit)
    old_stdout = sys.stdout
    sys.stdout = captured_output = OutputCapture()
    
    try:
        # Run the agent (same as Streamlit)
        response = agent_executor.run(user_query)
        verbose_output = captured_output.get_output()
        sys.stdout = old_stdout
        
        app.logger.debug(f"[CHAT] Verbose output length: {len(verbose_output)}")
        
        # Extract SQL query and results from verbose output
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
# API: Analytics Data
# --------------------------
@app.route('/api/analytics-data')
def api_analytics_data():
    """Pre-built analytics queries (no LLM needed)"""
    app.logger.info("Fetching analytics data...")
    conn = get_db_connection()
    data = {}
    
    try:
        # 1. Wins by Team
        df_wins = pd.read_sql_query("""
            SELECT winner as Team, COUNT(*) as Wins 
            FROM outcome 
            WHERE winner IS NOT NULL 
            GROUP BY winner 
            ORDER BY Wins DESC 
            LIMIT 10
        """, conn)
        data['wins_by_team'] = df_wins.to_dict(orient='records')
        
        # 2. Toss Decision
        df_toss = pd.read_sql_query("""
            SELECT toss_decision as Decision, COUNT(*) as Count 
            FROM toss 
            GROUP BY toss_decision
        """, conn)
        data['toss_decision'] = df_toss.to_dict(orient='records')
        
        # 3. Matches by Type
        df_type = pd.read_sql_query("""
            SELECT match_type as Type, COUNT(*) as Matches 
            FROM matches 
            GROUP BY match_type
        """, conn)
        data['matches_by_type'] = df_type.to_dict(orient='records')
        
        # 4. Top Run Scorers
        df_runs = pd.read_sql_query("""
            SELECT striker as Player, SUM(runs_off_bat) as Runs 
            FROM ball_by_ball 
            GROUP BY striker 
            ORDER BY Runs DESC 
            LIMIT 10
        """, conn)
        data['top_scorers'] = df_runs.to_dict(orient='records')
        
        # 5. Matches per Season
        df_season = pd.read_sql_query("""
            SELECT season as Season, COUNT(*) as Matches 
            FROM matches 
            WHERE season IS NOT NULL 
            GROUP BY season 
            ORDER BY season
        """, conn)
        data['matches_per_season'] = df_season.to_dict(orient='records')

        # 6. Win Method
        df_method = pd.read_sql_query("""
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
        """, conn)
        data['win_method'] = df_method.to_dict(orient='records')

        # 7. Top Wicket Takers
        df_wickets = pd.read_sql_query("""
            SELECT bowler as Player, COUNT(*) as Wickets 
            FROM ball_by_ball 
            WHERE wicket_type IS NOT NULL AND wicket_type != '' 
            GROUP BY bowler 
            ORDER BY Wickets DESC 
            LIMIT 10
        """, conn)
        data['top_wicket_takers'] = df_wickets.to_dict(orient='records')

        # 8. Matches by City
        df_city = pd.read_sql_query("""
            SELECT city as City, COUNT(*) as Matches 
            FROM matches 
            WHERE city IS NOT NULL 
            GROUP BY city 
            ORDER BY Matches DESC 
            LIMIT 10
        """, conn)
        data['matches_by_city'] = df_city.to_dict(orient='records')

    except Exception as e:
        app.logger.error(f"Error fetching analytics: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        
    return jsonify(data)


# --------------------------
# API: Schema Info
# --------------------------
@app.route('/api/schema-info')
def api_schema_info():
    """Return database schema information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
    tables = cursor.fetchall()
    
    schema = {}
    for table in tables:
        t_name = table['name']
        cursor.execute(f"PRAGMA table_info({t_name})")
        cols = cursor.fetchall()
        schema[t_name] = [{"name": c[1], "type": c[2]} for c in cols]
    
    conn.close()
    return jsonify(schema)

# --------------------------
# API: Query Runner
# --------------------------
@app.route('/api/query-runner/tables')
def api_query_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
    tables = [row['name'] for row in cursor.fetchall()]
    conn.close()
    return jsonify(tables)

@app.route('/api/query-runner/columns/<table_name>')
def api_query_columns(table_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        return jsonify(columns)
    except:
        return jsonify([])
    finally:
        conn.close()

@app.route('/api/query-runner/execute', methods=['POST'])
def api_query_execute():
    """Execute a safe parameterized query"""
    data = request.json
    table = data.get('table')
    columns = data.get('columns')
    filters = data.get('filters')
    limit = data.get('limit', 100)
    
    if not table or not columns:
        return jsonify({"error": "Invalid request"}), 400
        
    conn = get_db_connection()
    try:
        cols_str = ", ".join(columns)
        query = f"SELECT {cols_str} FROM {table}"
        params = []
        
        if filters:
            where_clauses = []
            for f in filters:
                col = f['column']
                op = f['operator']
                val = f['value']
                
                if op not in ['=', '>', '<', '>=', '<=', 'LIKE', '!=']:
                    continue
                    
                where_clauses.append(f"{col} {op} ?")
                params.append(val)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
        
        query += f" LIMIT {int(limit)}"
        
        df = pd.read_sql_query(query, conn, params=params)
        return jsonify({
            "columns": columns,
            "rows": df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --------------------------
# API: Match Predictor
# --------------------------
@app.route('/api/match-predictor/predict', methods=['POST'])
def api_match_predictor():
    """Match prediction using LLM with database context"""
    data = request.json
    team1 = data.get('team1', '')
    team2 = data.get('team2', '')
    venue = data.get('venue', '')
    
    if not llm:
        return jsonify({"error": "LLM not initialized"}), 500
    
    def generate():
        try:
            conn = get_db_connection()
            
            yield f"data: {json.dumps({'type': 'status', 'message': '📊 Analyzing head-to-head records...'})}\n\n"
            
            # Get head-to-head stats
            h2h_stats = pd.read_sql_query(f"""
                SELECT m.match_name, m.date, m.venue, o.winner
                FROM matches m
                JOIN outcome o ON m.match_id = o.match_id
                WHERE (m.match_name LIKE '%{team1}%' AND m.match_name LIKE '%{team2}%')
                ORDER BY m.date DESC LIMIT 10
            """, conn)
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'✅ Found {len(h2h_stats)} head-to-head matches'})}\n\n"
            
            # Get venue stats
            venue_stats = pd.read_sql_query(f"""
                SELECT o.winner, COUNT(*) as wins
                FROM matches m JOIN outcome o ON m.match_id = o.match_id
                WHERE m.venue LIKE '%{venue}%'
                GROUP BY o.winner ORDER BY wins DESC LIMIT 5
            """, conn)
            
            yield f"data: {json.dumps({'type': 'status', 'message': '✅ Venue statistics collected'})}\n\n"
            
            # Recent form
            team1_form = pd.read_sql_query(f"""
                SELECT m.date, o.winner FROM matches m
                JOIN outcome o ON m.match_id = o.match_id
                WHERE m.match_name LIKE '%{team1}%'
                ORDER BY m.date DESC LIMIT 5
            """, conn)
            
            team2_form = pd.read_sql_query(f"""
                SELECT m.date, o.winner FROM matches m
                JOIN outcome o ON m.match_id = o.match_id
                WHERE m.match_name LIKE '%{team2}%'
                ORDER BY m.date DESC LIMIT 5
            """, conn)
            
            conn.close()
            
            yield f"data: {json.dumps({'type': 'status', 'message': '🤖 AI is generating prediction...'})}\n\n"
            
            prompt = f"""You are an expert cricket analyst. Predict the outcome of {team1} vs {team2} at {venue}.

Head-to-Head: {h2h_stats.to_string() if not h2h_stats.empty else 'No data'}
Venue Stats: {venue_stats.to_string() if not venue_stats.empty else 'No data'}
{team1} Recent: {team1_form.to_string() if not team1_form.empty else 'No data'}
{team2} Recent: {team2_form.to_string() if not team2_form.empty else 'No data'}

Provide win probabilities, key factors, and prediction."""

            # Stream response
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
# API: Player Analysis
# --------------------------
@app.route('/api/player-comparison')
def api_player_comparison():
    """Compare two players' stats"""
    p1 = request.args.get('p1')
    p2 = request.args.get('p2')
    
    if not p1 or not p2:
        return jsonify({"error": "Missing player names"}), 400
        
    conn = get_db_connection()
    try:
        batting_df = pd.read_sql_query("""
            SELECT striker as player,
                   SUM(runs_off_bat) as runs,
                   COUNT(DISTINCT match_id) as matches,
                   COUNT(*) as balls,
                   ROUND(SUM(runs_off_bat) * 100.0 / COUNT(*), 2) as strike_rate
            FROM ball_by_ball
            WHERE striker IN (?, ?)
            GROUP BY striker
        """, conn, params=(p1, p2))
        
        bowling_df = pd.read_sql_query("""
            SELECT bowler as player,
                   COUNT(*) as balls,
                   SUM(runs_off_bat) as runs_conceded,
                   COUNT(CASE WHEN wicket_type IS NOT NULL AND wicket_type != '' THEN 1 END) as wickets,
                   ROUND(SUM(runs_off_bat) * 6.0 / COUNT(*), 2) as economy
            FROM ball_by_ball
            WHERE bowler IN (?, ?)
            GROUP BY bowler
        """, conn, params=(p1, p2))
        
        return jsonify({
            "batting": batting_df.to_dict(orient='records'),
            "bowling": bowling_df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/batter-vs-bowler')
def api_batter_vs_bowler():
    """Head-to-head stats between batter and bowler"""
    batter = request.args.get('batter')
    bowler = request.args.get('bowler')
    
    if not batter or not bowler:
        return jsonify({"error": "Missing batter or bowler name"}), 400
        
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as balls_faced,
                SUM(runs_off_bat) as runs_scored,
                COUNT(CASE WHEN wicket_type IS NOT NULL AND wicket_type != '' AND player_dismissed = ? THEN 1 END) as dismissals,
                SUM(CASE WHEN runs_off_bat = 4 THEN 1 ELSE 0 END) as fours,
                SUM(CASE WHEN runs_off_bat = 6 THEN 1 ELSE 0 END) as sixes
            FROM ball_by_ball
            WHERE striker = ? AND bowler = ?
        """, conn, params=(batter, batter, bowler))
        result = df.to_dict(orient='records')[0] if not df.empty else {}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/player-venue')
def api_player_venue():
    """Player performance by venue"""
    player = request.args.get('player')
    if not player:
        return jsonify({"error": "Missing player name"}), 400
        
    conn = get_db_connection()
    try:
        df = pd.read_sql_query("""
            SELECT 
                m.venue,
                SUM(b.runs_off_bat) as runs,
                COUNT(b.id) as balls,
                ROUND(SUM(b.runs_off_bat) * 100.0 / COUNT(b.id), 2) as strike_rate,
                COUNT(DISTINCT m.match_id) as matches
            FROM ball_by_ball b
            JOIN matches m ON b.match_id = m.match_id
            WHERE b.striker = ?
            GROUP BY m.venue
            ORDER BY runs DESC
            LIMIT 10
        """, conn, params=(player,))
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/player-spin-pace')
def api_player_spin_pace():
    """Analyze player performance vs spin/pace using LLM classification"""
    player = request.args.get('player')
    if not player:
        return jsonify({"error": "Missing player name"}), 400
        
    conn = get_db_connection()
    try:
        # Get bowlers faced
        bowlers_df = pd.read_sql_query("""
            SELECT DISTINCT bowler, COUNT(*) as balls
            FROM ball_by_ball WHERE striker = ?
            GROUP BY bowler ORDER BY balls DESC LIMIT 20
        """, conn, params=(player,))
        bowlers_list = bowlers_df['bowler'].tolist()
        
        if not bowlers_list:
            return jsonify({"error": "No data found for player"}), 404

        if not llm:
            return jsonify({"error": "LLM not initialized"}), 500
             
        # Classify bowlers
        prompt = f"""Classify these cricket bowlers as 'Spin' or 'Pace'. Return ONLY a JSON object.
Bowlers: {', '.join(bowlers_list)}"""
        
        response = llm.invoke(prompt)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        classification = json.loads(content.strip())
        
        # Aggregate stats
        stats_df = pd.read_sql_query(f"""
            SELECT bowler, SUM(runs_off_bat) as runs, COUNT(*) as balls
            FROM ball_by_ball
            WHERE striker = ? AND bowler IN ({','.join(['?']*len(bowlers_list))})
            GROUP BY bowler
        """, conn, params=[player] + bowlers_list)
        
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
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --------------------------
# Run Application
# --------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
