<div align="center">

# 🏏 AI Cricket Analytics Hub

<strong>Natural Language → SQL → Insight.</strong><br>
Ask cricket questions, see the exact SQL generated, view raw database results, and explore rich dashboards – all in a clean, blue-themed Streamlit interface powered by modern LLM tooling.

</div>

---

## 🎯 What Is This Project?
The AI Cricket Analytics Hub is an interactive analytics and exploration application for structured cricket match data stored in a local SQLite database (`cricket_matches.db`). It combines:

- 💬 An AI chat assistant (Gemini + LangChain SQL Agent) that converts plain English questions into SQL queries
- 🧠 Transparent query generation: you always see the full SQL and raw output
- 📈 Pre-built analytical dashboards (wins, toss impact, match types, venues, trends)
- 🗄️ Schema explorer: introspect table structures directly
- 🎨 Clean, responsive light & dark blue UI with emojis for clarity and delight

> Note: This README focuses on the application layer (interactive analytics + AI querying). Data population/ETL is assumed to be handled separately.

---

## ✨ Key Features
- 🤖 **LLM-Powered Querying:** Gemini 2.0 Flash Experimental (`gemini-2.0-flash-exp`) via LangChain
- 🔍 **Full Transparency:** Parsed AI answer + generated SQL + raw SQL result block
- 📊 **Dashboards:** Plotly visualizations (bar, pie, trend) with a unified theme
- 🗃️ **Schema Inspection:** Run `check_schema.py` to list tables & columns
- 🚀 **Fast Setup:** Single-command launch with Streamlit
- 🛡️ **Safe Execution:** Read-only style querying (design encourages inspection before modification)

---

## 🧩 Architecture Overview
```
User → Streamlit UI → LangChain SQL Agent → Gemini LLM
                                 ↓
                           SQLite Database
                                 ↓
                     Results + Generated SQL → UI
```

Core Flow:
1. User enters a natural language question.
2. The LangChain SQL agent (Gemini backend) plans & generates SQL.
3. SQL is executed against `cricket_matches.db`.
4. Three outputs are displayed: human answer, SQL text, raw result rows.
5. Dashboards use direct curated SQL queries (not LLM-generated) for reliability.

---

## 🛠 Tech Stack
| Layer              | Tools / Libraries |
|--------------------|------------------|
| Frontend UI        | Streamlit, Custom CSS (gradient light/dark blue theme) |
| Visualization      | Plotly Express, Plotly Graph Objects |
| Data Access        | SQLite (local file) |
| LLM / Orchestration| LangChain, `langchain_google_genai` |
| Model              | Gemini 2.5 Flash Exp (zero-temperature deterministic querying) |
| Utilities          | `python-dotenv`, `pandas`, `datetime` |

Environment Variables (in `.env`):
- `GEMINI_API_KEY` – Google Generative AI key for Gemini
- Optional: `OPENAI_API_KEY`, `GROQ_API_KEY`, `HF_TOKEN` (present but not required for core app)

---

## 📂 Code Overview
Only two core Python files are required for the running application:

| File | Purpose |
|------|---------|
| `app.py` | Main multi-page Streamlit application (chat, dashboards, schema viewer) |
| `check_schema.py` | Utility script to print table names and column definitions from the SQLite DB |

> The application intentionally avoids referencing any other module folders to keep the runtime surface minimal.

### `app.py` Highlights
- Page navigation: Home, AI Chat Assistant, Standard Analytics, Database Schema
- Custom CSS theme injection for cohesive visuals
- Function `run_agent_with_query_capture` captures stdout from LangChain verbose run to extract:
  - Generated SQL (`Action Input:` parsing)
  - Raw observation block (query result rows)
- Chat history stored in `st.session_state` with timestamps
- Plotly charts use carefully selected blue gradients and white card backgrounds

### `check_schema.py` Logic
```python
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")  # list tables
cursor.execute(f'PRAGMA table_info({table_name})')                     # introspect columns
```

---

## 🚀 Setup Instructions (Windows PowerShell)

```powershell
# 1. Clone repository
git clone https://github.com/NinadGawali/AI_Cricket_Analysis.git
cd AI_Cricket_Analysis

# 2. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment variables
# Edit .env and ensure GEMINI_API_KEY is present.

# 5. (Optional) Inspect database schema
python check_schema.py

# 6. Launch the app
streamlit run app.py
```

### macOS / Linux Variant
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🗄️ Database Expectations
The SQLite file `cricket_matches.db` should exist in the project root before launching. If not present:
- Provide or generate it externally (ETL pipeline not included here)
- Use `check_schema.py` to validate tables

Recommended Core Tables (as inferred from analytics queries in `app.py`):
- `matches(match_id, match_type, venue, date, ...)`
- `outcome(winner, ...)`
- `toss(toss_decision, ...)`
- `players(player_name, ...)`

> Ensure foreign keys and indices are added as needed for performance.

---

## 💬 AI Chat Usage
1. Navigate to “AI Chat Assistant” page.
2. Enter a natural language question (e.g. *"Which team has won the most matches?"*).
3. Upon execution you will see:
   - Parsed assistant answer (LLM synthesized)
   - Generated SQL (exact string sent to DB)
   - Raw SQL output (tabular rows as text)
4. Use this transparency to refine further questions.

---

## 📊 Analytics Overview
Dashboards execute curated SQL queries (not LLM-generated) for reliability. Current visual modules:
- 🏆 Top Teams by Wins
- ⚡ Toss Decision Distribution
- 🎮 Matches by Type
- 👤 Player / Performance Trends (extensible)
All charts use Plotly with white backgrounds + blue gradients for visual clarity.

---

## 🔐 Environment & Security Notes
- Keep `.env` out of version control (add to `.gitignore` if not already)
- Never expose API keys in screenshots or logs
- Consider rotating API keys periodically

---

## 🧪 Extensibility Suggestions
- Add caching for repeated analytical queries
- Introduce role-based access if deployed publicly
- Add export buttons (CSV/Excel) for raw query results
- Integrate model comparison (Gemini vs. Groq vs. OpenAI) behind a toggle

---

## 🛠 Troubleshooting
| Issue | Possible Fix |
|-------|--------------|
| `GEMINI_API_KEY not found` | Verify `.env` file + restart terminal |
| Blank analytics charts | Confirm tables exist in `cricket_matches.db` |
| SQL errors in chat output | Refine question; ensure schema matches expected columns |
| Import error `langchain_google_genai` | Reinstall requirements: `pip install -r requirements.txt` |

---

## 🎥 Sample Video (Placeholder)
> Embed or link a demo here (e.g. Loom / GitHub asset). Replace below once recorded.

`[ Place demonstration video / GIF here ]`

You can also attach via: 
```
https://github.com/user-attachments/assets/<your-asset-id>
```

---

## 📜 License
Specify licensing terms here (MIT / Apache-2.0 / Proprietary). *Currently unspecified.*

---

## 🤝 Contributions
Feel free to open issues or PRs for:
- New visualizations
- Query optimization
- UI polish enhancements
- Localization / internationalization

---

## ✅ Quick Start (TL;DR)
```powershell
git clone https://github.com/NinadGawali/AI_Cricket_Analysis.git
cd AI_Cricket_Analysis
python -m venv .venv; .\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Enjoy exploring cricket data intelligently! 🏏🤖

