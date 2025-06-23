#!/bin/bash
# apps/semantic_search/app.sh

# Domino App Launcher for Semantic Search Engine
# This script launches the Streamlit application in Domino

echo "Starting Quality Analytics Semantic Search Engine..."

# Set environment variables
export STREAMLIT_SERVER_PORT=8888
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Navigate to app directory
cd /mnt/code/apps/semantic_search

# Launch Streamlit app
echo "Launching Streamlit on port 8888..."
streamlit run app.py \
    --server.port=8888 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --logger.level=info