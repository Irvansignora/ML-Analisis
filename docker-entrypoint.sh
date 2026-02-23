#!/bin/bash
# Docker Entrypoint Script for Sales ML System
# ============================================

set -e

# Function to wait for a service
wait_for_service() {
    local host=$1
    local port=$2
    local timeout=${3:-30}
    
    echo "Waiting for $host:$port..."
    for i in $(seq 1 $timeout); do
        if nc -z $host $port 2>/dev/null; then
            echo "$host:$port is available!"
            return 0
        fi
        sleep 1
    done
    echo "Timeout waiting for $host:$port"
    return 1
}

# Function to run Streamlitun_streamlit() {
    echo "Starting Streamlit Dashboard..."
    exec streamlit run app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --browser.gatherUsageStats=false
}

# Function to run API
run_api() {
    echo "Starting Flask API Server..."
    exec gunicorn \
        --bind 0.0.0.0:${API_PORT:-5000} \
        --workers 4 \
        --threads 2 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile - \
        --preload \
        api:app
}

# Function to run both services
run_both() {
    echo "Starting both Streamlit and API..."
    
    # Start API in background
    gunicorn \
        --bind 0.0.0.0:${API_PORT:-5000} \
        --workers 2 \
        --threads 2 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile - \
        --daemon \
        api:app
    
    # Wait for API to be ready
    sleep 3
    
    # Start Streamlit in foreground
    exec streamlit run app.py \
        --server.port=${PORT:-8501} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --browser.gatherUsageStats=false
}

# Function to run Jupyter (development mode)
run_jupyter() {
    echo "Starting Jupyter Notebook..."
    exec jupyter notebook \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password=''
}

# Main execution
case "${1:-streamlit}" in
    streamlit)
        run_streamlit
        ;;
    api)
        run_api
        ;;
    both)
        run_both
        ;;
    jupyter)
        run_jupyter
        ;;
    bash|sh)
        exec "$@"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Usage: $0 {streamlit|api|both|jupyter|bash}"
        exit 1
        ;;
esac
