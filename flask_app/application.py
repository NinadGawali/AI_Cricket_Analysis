import os
import sys
import sqlite3
import json
import logging
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

application = Flask(__name__)
app = application  # Elastic Beanstalk expects 'application'

# Enable verbose logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
app.logger.setLevel(logging.INFO)

# Configuration - Use environment variable or default path
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "cricket_matches.db"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not os.path.exists(DB_PATH):
    app.logger.warning(f"Database not found at {DB_PATH}")

# Initialize LLM and Agent
llm = None
agent_executor = None
db = None

def init_agent():
    global llm, agent_executor, db
    if not GEMINI_API_KEY:
        app.logger.error("GEMINI_API_KEY not found")
        return

    try:
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GEMINI_API_KEY,
            temperature=0,
            max_output_tokens=2048,
            streaming=True
        )
        app.logger.info("[LLM] Initialized ChatGoogleGenerativeAI with model gemini-1.5-flash")
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        app.logger.info("Agent initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing agent: {e}")

init_agent()

# Helper for capturing output
class OutputCapture:
    def __init__(self):
        self.output = []
    def write(self, text):
        self.output.append(text)
    def flush(self):
        pass
    def get_output(self):
        return ''.join(self.output)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Routes
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

# Health check endpoint for EB
@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

# API Endpoints

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    user_query = data.get('message')
    
    if not agent_executor:
        return jsonify({"error": "Agent not initialized"}), 500

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = OutputCapture()
    
    try:
        response = agent_executor.run(user_query)
        verbose_output = captured_output.get_output()
        sys.stdout = old_stdout
        
        # Parse SQL and Result (Reuse logic from Streamlit app)
        sql_query = None
        sql_result = None
        
        if "Action Input:" in verbose_output:
            lines = verbose_output.split('\n')
            for i, line in enumerate(lines):
                if "Action Input:" in line:
                    query_lines = []
                    for j in range(i+1, min(i+20, len(lines))):
                        if "Observation:" in lines[j]:
                            break
                        if lines[j].strip() and "Action:" not in lines[j]:
                            query_lines.append(lines[j].strip())
                    sql_query = ' '.join(query_lines).replace('"', '').replace("'", "").strip()
                    
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
        
        return jsonify({
            "response": response,
            "sql_query": sql_query,
            "sql_result": sql_result
        })
        
    except Exception as e:
        sys.stdout = old_stdout
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics-data')
def api_analytics_data():
    app.logger.info("Fetching analytics data...")
    conn = get_db_connection()
    data = {}
    
    try:
        # 1. Wins by Team
        df_wins = pd.read_sql_query("SELECT winner as Team, COUNT(*) as Wins FROM outcome WHERE winner IS NOT NULL GROUP BY winner ORDER BY Wins DESC LIMIT 10", conn)
        data['wins_by_team'] = df_wins.to_dict(orient='records')
        
        # 2. Toss Decision
        df_toss = pd.read_sql_query("SELECT toss_decision as Decision, COUNT(*) as Count FROM toss GROUP BY toss_decision", conn)
        data['toss_decision'] = df_toss.to_dict(orient='records')
        
        # 3. Matches by Type
        df_type = pd.read_sql_query("SELECT match_type as Type, COUNT(*) as Matches FROM matches GROUP BY match_type", conn)
        data['matches_by_type'] = df_type.to_dict(orient='records')
        
        # 4. Top Run Scorers
        df_runs = pd.read_sql_query("SELECT striker as Player, SUM(runs_off_bat) as Runs FROM ball_by_ball GROUP BY striker ORDER BY Runs DESC LIMIT 10", conn)
        data['top_scorers'] = df_runs.to_dict(orient='records')
        
        # 5. Matches per Season
        df_season = pd.read_sql_query("SELECT season as Season, COUNT(*) as Matches FROM matches WHERE season IS NOT NULL GROUP BY season ORDER BY season", conn)
        data['matches_per_season'] = df_season.to_dict(orient='records')

        # 6. Win Method
        df_method = pd.read_sql_query("SELECT CASE WHEN won_by_runs > 0 THEN 'By Runs' WHEN won_by_wickets > 0 THEN 'By Wickets' ELSE 'Other' END as Method, COUNT(*) as Count FROM outcome WHERE winner IS NOT NULL GROUP BY Method", conn)
        data['win_method'] = df_method.to_dict(orient='records')

        # 7. Top Wicket Takers
        df_wickets = pd.read_sql_query("SELECT bowler as Player, COUNT(*) as Wickets FROM ball_by_ball WHERE wicket_type IS NOT NULL AND wicket_type != '' GROUP BY bowler ORDER BY Wickets DESC LIMIT 10", conn)
        data['top_wicket_takers'] = df_wickets.to_dict(orient='records')

        # 8. Matches by City
        df_city = pd.read_sql_query("SELECT city as City, COUNT(*) as Matches FROM matches WHERE city IS NOT NULL GROUP BY city ORDER BY Matches DESC LIMIT 10", conn)
        data['matches_by_city'] = df_city.to_dict(orient='records')

    except Exception as e:
        app.logger.error(f"Error fetching analytics: {e}")
    finally:
        conn.close()
        
    return jsonify(data)


@app.route('/api/schema-info')
def api_schema_info():
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
    data = request.json
    table = data.get('table')
    columns = data.get('columns') # List of columns
    filters = data.get('filters') # List of {column, operator, value}
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
                op = f['operator'] # =, >, <, LIKE
                val = f['value']
                
                if op not in ['=', '>', '<', '>=', '<=', 'LIKE', '!=']:
                    continue
                    
                where_clauses.append(f"{col} {op} ?")
                params.append(val)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
        
        query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        return jsonify({
            "columns": columns,
            "rows": df.to_dict(orient='records') # List of dicts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# Match Predictor Agent API
@app.route('/api/match-predictor/predict', methods=['POST'])
def api_match_predictor():
    data = request.json
    team1 = data.get('team1', '')
    team2 = data.get('team2', '')
    venue = data.get('venue', '')
    
    if not llm:
        return jsonify({"error": "LLM not initialized"}), 500
    
    def generate():
        try:
            conn = get_db_connection()
            app.logger.debug("[Predictor] Request: team1=%s team2=%s venue=%s", team1, team2, venue)
            
            yield f"data: {json.dumps({'type': 'status', 'message': '📊 Analyzing head-to-head records...'})}\n\n"
            
            # Get head-to-head stats
            h2h_query = f"""
                SELECT 
                    m.match_name,
                    m.date,
                    m.venue,
                    o.winner
                FROM matches m
                JOIN outcome o ON m.match_id = o.match_id
                WHERE (m.match_name LIKE '%{team1}%' AND m.match_name LIKE '%{team2}%')
                   OR (m.match_name LIKE '%{team2}%' AND m.match_name LIKE '%{team1}%')
                ORDER BY m.date DESC
                LIMIT 10
            """
            h2h_stats = pd.read_sql_query(h2h_query, conn)
            app.logger.debug("[Predictor] h2h rows=%d", len(h2h_stats))
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'✅ Found {len(h2h_stats)} head-to-head matches'})}\n\n"
            
            # Get venue stats
            venue_query = f"""
                SELECT 
                    o.winner,
                    COUNT(*) as wins
                FROM matches m
                JOIN outcome o ON m.match_id = o.match_id
                WHERE m.venue LIKE '%{venue}%'
                GROUP BY o.winner
                ORDER BY wins DESC
                LIMIT 5
            """
            venue_stats = pd.read_sql_query(venue_query, conn)
            app.logger.debug("[Predictor] venue_stats rows=%d", len(venue_stats))
            
            yield f"data: {json.dumps({'type': 'status', 'message': '✅ Venue statistics collected'})}\n\n"
            
            # Get recent form
            team1_form = pd.read_sql_query(f"""
                SELECT m.date, o.winner, o.won_by_runs, o.won_by_wickets
                FROM matches m
                JOIN outcome o ON m.match_id = o.match_id
                WHERE m.match_name LIKE '%{team1}%'
                ORDER BY m.date DESC
                LIMIT 5
            """, conn)
            app.logger.debug("[Predictor] %s form rows=%d", team1, len(team1_form))
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'✅ {team1} recent form analyzed'})}\n\n"
            
            team2_form = pd.read_sql_query(f"""
                SELECT m.date, o.winner, o.won_by_runs, o.won_by_wickets
                FROM matches m
                JOIN outcome o ON m.match_id = o.match_id
                WHERE m.match_name LIKE '%{team2}%'
                ORDER BY m.date DESC
                LIMIT 5
            """, conn)
            app.logger.debug("[Predictor] %s form rows=%d", team2, len(team2_form))
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'✅ {team2} recent form analyzed'})}\n\n"
            
            conn.close()
            
            yield f"data: {json.dumps({'type': 'status', 'message': '🤖 AI is generating prediction...'})}\n\n"
            
            # Create prediction prompt
            prompt = f"""You are an expert cricket match analyst. Predict the outcome of this match with detailed analysis.

**Match Details:**
- Team 1: {team1}
- Team 2: {team2}
- Venue: {venue}

**Head-to-Head Record:**
{h2h_stats.to_string() if not h2h_stats.empty else 'No previous matches found'}

**Venue Statistics:**
{venue_stats.to_string() if not venue_stats.empty else 'Limited venue data available'}

**Recent Form - {team1}:**
{team1_form.to_string() if not team1_form.empty else 'No recent matches found'}

**Recent Form - {team2}:**
{team2_form.to_string() if not team2_form.empty else 'No recent matches found'}

**Task:** Provide a comprehensive match prediction including win probabilities, key factors, and strategic insights.
"""

            # Stream the LLM response
            full_response = ""
            chunk_count = 0
            try:
                for chunk in llm.stream(prompt):
                    # Handle different chunk formats
                    content = ""
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                    elif isinstance(chunk, str):
                        content = chunk
                    elif hasattr(chunk, 'text'):
                        content = chunk.text
                    
                    if content:
                        full_response += content
                        chunk_count += 1
                        if chunk_count % 20 == 0:
                            app.logger.debug("[Predictor] Streamed %d chunks so far", chunk_count)
                        yield f"data: {json.dumps({'type': 'content', 'text': content})}\n\n"
            except Exception as stream_error:
                # Fallback to non-streaming if streaming fails
                app.logger.exception("[Predictor] Streaming failed, falling back to non-streaming: %s", stream_error)
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating complete response...'})}\n\n"
                fallback_obj = llm.invoke(prompt)
                full_response = fallback_obj.content if hasattr(fallback_obj, 'content') else str(fallback_obj)
                app.logger.debug("[Predictor] Fallback content length=%d", len(full_response) if isinstance(full_response, str) else -1)
                if isinstance(full_response, str) and len(full_response) > 0:
                    chunk_size = 800
                    for i in range(0, len(full_response), chunk_size):
                        part = full_response[i:i+chunk_size]
                        yield f"data: {json.dumps({'type': 'content', 'text': part})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No content generated by LLM'})}\n\n"
            
            # Send statistics at the end
            yield f"data: {json.dumps({'type': 'stats', 'data': {'head_to_head': h2h_stats.to_dict(orient='records') if not h2h_stats.empty else [], 'venue_stats': venue_stats.to_dict(orient='records') if not venue_stats.empty else [], 'team1_form': team1_form.to_dict(orient='records') if not team1_form.empty else [], 'team2_form': team2_form.to_dict(orient='records') if not team2_form.empty else []}})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

# Player Analysis APIs

@app.route('/api/player-comparison')
def api_player_comparison():
    p1 = request.args.get('p1')
    p2 = request.args.get('p2')
    
    if not p1 or not p2:
        return jsonify({"error": "Missing player names"}), 400
        
    conn = get_db_connection()
    try:
        # Batting Stats
        batting_query = """
            SELECT striker as player,
                   SUM(runs_off_bat) as runs,
                   COUNT(DISTINCT match_id) as matches,
                   COUNT(*) as balls,
                   ROUND(SUM(runs_off_bat) * 100.0 / COUNT(*), 2) as strike_rate,
                   ROUND(SUM(runs_off_bat) * 1.0 / NULLIF((
                       SELECT COUNT(*) FROM ball_by_ball b2 
                       WHERE b2.player_dismissed = ball_by_ball.striker
                   ), 0), 2) as average
            FROM ball_by_ball
            WHERE striker IN (?, ?)
            GROUP BY striker
        """
        batting_df = pd.read_sql_query(batting_query, conn, params=(p1, p2))
        
        # Bowling Stats
        bowling_query = """
            SELECT bowler as player,
                   COUNT(*) as balls,
                   SUM(runs_off_bat) as runs_conceded,
                   COUNT(CASE WHEN wicket_type IS NOT NULL AND wicket_type != '' THEN 1 END) as wickets,
                   ROUND(SUM(runs_off_bat) * 6.0 / COUNT(*), 2) as economy
            FROM ball_by_ball
            WHERE bowler IN (?, ?)
            GROUP BY bowler
        """
        bowling_df = pd.read_sql_query(bowling_query, conn, params=(p1, p2))
        
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
    batter = request.args.get('batter')
    bowler = request.args.get('bowler')
    
    if not batter or not bowler:
        return jsonify({"error": "Missing batter or bowler name"}), 400
        
    conn = get_db_connection()
    try:
        query = """
            SELECT 
                COUNT(*) as balls_faced,
                SUM(runs_off_bat) as runs_scored,
                COUNT(CASE WHEN wicket_type IS NOT NULL AND wicket_type != '' AND player_dismissed = ? THEN 1 END) as dismissals,
                SUM(CASE WHEN runs_off_bat = 4 THEN 1 ELSE 0 END) as fours,
                SUM(CASE WHEN runs_off_bat = 6 THEN 1 ELSE 0 END) as sixes
            FROM ball_by_ball
            WHERE striker = ? AND bowler = ?
        """
        df = pd.read_sql_query(query, conn, params=(batter, batter, bowler))
        result = df.to_dict(orient='records')[0] if not df.empty else {}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/player-venue')
def api_player_venue():
    player = request.args.get('player')
    if not player:
        return jsonify({"error": "Missing player name"}), 400
        
    conn = get_db_connection()
    try:
        query = """
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
        """
        df = pd.read_sql_query(query, conn, params=(player,))
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/player-spin-pace')
def api_player_spin_pace():
    player = request.args.get('player')
    if not player:
        return jsonify({"error": "Missing player name"}), 400
        
    conn = get_db_connection()
    try:
        # 1. Get list of bowlers faced
        bowlers_query = """
            SELECT DISTINCT bowler, COUNT(*) as balls
            FROM ball_by_ball
            WHERE striker = ?
            GROUP BY bowler
            ORDER BY balls DESC
            LIMIT 20
        """
        bowlers_df = pd.read_sql_query(bowlers_query, conn, params=(player,))
        bowlers_list = bowlers_df['bowler'].tolist()
        
        if not bowlers_list:
             return jsonify({"error": "No data found for player"}), 404

        # 2. Classify bowlers using LLM
        if not llm:
             return jsonify({"error": "LLM not initialized"}), 500
             
        prompt = f"""Classify the following cricket bowlers as 'Spin' or 'Pace'. 
        Return ONLY a JSON object where keys are bowler names and values are 'Spin' or 'Pace'.
        Bowlers: {', '.join(bowlers_list)}"""
        
        response = llm.invoke(prompt)
        content = response.content
        # Clean up markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        classification = json.loads(content)
        
        # 3. Aggregate stats based on classification
        spin_runs = 0
        spin_balls = 0
        pace_runs = 0
        pace_balls = 0
        
        # We need to query stats for each bowler to aggregate correctly
        stats_query = f"""
            SELECT bowler, SUM(runs_off_bat) as runs, COUNT(*) as balls
            FROM ball_by_ball
            WHERE striker = ? AND bowler IN ({','.join(['?']*len(bowlers_list))})
            GROUP BY bowler
        """
        stats_df = pd.read_sql_query(stats_query, conn, params=[player] + bowlers_list)
        
        for _, row in stats_df.iterrows():
            b_name = row['bowler']
            b_type = classification.get(b_name, 'Unknown')
            if b_type == 'Spin':
                spin_runs += row['runs']
                spin_balls += row['balls']
            elif b_type == 'Pace':
                pace_runs += row['runs']
                pace_balls += row['balls']
                
        return jsonify({
            "spin": {"runs": int(spin_runs), "balls": int(spin_balls), "avg_sr": round(spin_runs*100/spin_balls, 2) if spin_balls else 0},
            "pace": {"runs": int(pace_runs), "balls": int(pace_balls), "avg_sr": round(pace_runs*100/pace_balls, 2) if pace_balls else 0},
            "details": classification
        })
        
    except Exception as e:
        app.logger.error(f"Error in spin/pace analysis: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    # For local development
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
