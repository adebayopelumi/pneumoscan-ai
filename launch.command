#!/bin/bash
# Double-click this file on Mac to launch PneumoScan AI

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "Starting PneumoScan AI..."

# Open browser after 3 seconds
(sleep 3 && open http://localhost:8501) &

# Start Streamlit
streamlit run app/streamlit_app.py --server.headless true
