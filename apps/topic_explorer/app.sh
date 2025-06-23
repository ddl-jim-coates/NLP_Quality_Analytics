#!/bin/bash
# apps/topic_explorer/app.sh

# Domino App Launcher for Topic Explorer
echo "Starting Quality Analytics Topic Explorer..."

# Set environment variables
export STREAMLIT_SERVER_PORT=8888
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Navigate to app directory
cd /mnt/code/apps/topic_explorer

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
echo "Launching Topic Explorer on port 8888..."
streamlit run app.py \
    --server.port=8888 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --logger.level=info