# Cricket Analytics Platform

A web-based cricket analytics platform built with Flask that provides natural language querying of cricket statistics, player analysis, match predictions, and interactive visualizations.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start with Docker](#quick-start-with-docker)
- [Local Development Setup](#local-development-setup)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Deployment](#deployment)

## Features

- **NLP to SQL Agent**: Query cricket statistics using natural language. The system converts questions to SQL and returns results.
- **Player Analysis**: Compare players head-to-head, analyze batter vs bowler matchups, and explore venue-specific performance.
- **Match Predictor**: AI-powered match outcome predictions based on historical data, head-to-head records, and venue statistics.
- **Interactive Analytics**: Pre-built dashboards with charts for team wins, top scorers, wicket takers, and seasonal trends.
- **Query Runner**: Visual query builder for direct database exploration without writing SQL.
- **Schema Browser**: View database structure and table relationships.

## Architecture

- **Backend**: Flask (Python)
- **Database**: SQLite
- **AI/LLM**: Google Gemini via LangChain
- **Frontend**: HTML, CSS, JavaScript with Plotly.js for charts
- **Deployment**: Docker, AWS Elastic Beanstalk

## Quick Start with Docker

### Pull and Run

```bash
# Pull the image from Docker Hub
docker pull ninad2004/cricket-analytics:latest

# Run with your Gemini API key
docker run -d -p 5000:5000 \
  -e GEMINI_API_KEY=your_gemini_api_key \
  ninad2004/cricket-analytics:latest
```

Access the application at http://localhost:5000

### Docker Commands Reference

```bash
# Check running containers
docker ps

# View logs
docker logs <container_id>

# Stop container
docker stop <container_id>

# Run with auto-restart
docker run -d -p 5000:5000 --restart unless-stopped \
  -e GEMINI_API_KEY=your_api_key \
  ninad2004/cricket-analytics:latest
```

## Local Development Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Google Gemini API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NinadGawali/AI_Cricket_Analysis.git
   cd AI_Cricket_Analysis/flask_app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   # Create .env file in flask_app directory
   echo "GEMINI_API_KEY=your_gemini_api_key" > .env
   ```

5. Run the application:
   ```bash
   python application_fixed.py
   ```

   The application will be available at http://localhost:5000

### Running with Gunicorn (Production)

```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 application_fixed:app
```

## API Endpoints

### Pages

| Route | Description |
|-------|-------------|
| `/` | Home page |
| `/chat` | NLP to SQL agent interface |
| `/analytics` | Analytics dashboard |
| `/player-analysis` | Player comparison and analysis |
| `/match-predictor` | Match prediction tool |
| `/query-runner` | Visual query builder |
| `/schema` | Database schema viewer |
| `/health` | Health check endpoint |

### REST APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Process natural language query |
| GET | `/api/analytics-data` | Retrieve analytics data |
| GET | `/api/schema-info` | Get database schema |
| GET | `/api/query-runner/tables` | List available tables |
| GET | `/api/query-runner/columns/<table>` | Get columns for a table |
| POST | `/api/query-runner/execute` | Execute a query |
| GET | `/api/player-comparison` | Compare two players |
| GET | `/api/batter-vs-bowler` | Head-to-head analysis |
| GET | `/api/player-venue` | Player venue statistics |
| GET | `/api/player-spin-pace` | Player performance vs spin/pace |
| POST | `/api/match-predictor/predict` | Generate match prediction |

## Project Structure

```
AI_Cricket_Analysis/
├── flask_app/                    # Main application
│   ├── application_fixed.py      # Flask application entry point
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile               # Docker configuration
│   ├── cricket_matches.db       # SQLite database
│   ├── .env                     # Environment variables
│   ├── templates/               # HTML templates
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── chat.html
│   │   ├── analytics.html
│   │   ├── player_analysis.html
│   │   ├── match_predictor.html
│   │   ├── query_runner.html
│   │   └── schema.html
│   └── static/                  # Static assets
│       ├── css/
│       └── js/
├── cricket_data/                # Legacy data module
├── app.py                       # Streamlit app (legacy)
└── README.md
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key for LLM features |
| `DB_PATH` | No | Custom path to SQLite database (default: `./cricket_matches.db`) |

### Obtaining a Gemini API Key

1. Go to Google AI Studio at https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and add it to your `.env` file

## Deployment

### AWS EC2 with Docker

1. SSH into your EC2 instance:
   ```bash
   ssh -i your-key.pem ec2-user@your-ec2-ip
   ```

2. Install Docker:
   ```bash
   sudo yum update -y
   sudo yum install docker -y
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```

3. Pull and run the container:
   ```bash
   docker pull ninad2004/cricket-analytics:latest
   
   docker run -d -p 80:5000 --restart unless-stopped \
     -e GEMINI_API_KEY=your_api_key \
     ninad2004/cricket-analytics:latest
   ```

4. Access via EC2 public IP at http://your-ec2-public-ip

## Database Schema

The application uses a SQLite database with the following tables:

- `matches` - Match metadata (date, venue, teams, format)
- `ball_by_ball` - Ball-by-ball delivery data
- `outcome` - Match results
- `toss` - Toss information
- `players` - Player registry
- `teams` - Team information

## License

This project is for educational and demonstration purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

