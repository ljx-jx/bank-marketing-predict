#!/bin/bash
# Start health-check server in background, then launch Streamlit.
set -e

python src/health.py &
HEALTH_PID=$!

# Wait for health server to be ready
sleep 1

# Start Streamlit (foreground)
streamlit run src/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

# If Streamlit exits, also stop the health server
kill $HEALTH_PID 2>/dev/null || true
