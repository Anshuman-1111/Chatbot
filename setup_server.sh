#!/bin/bash

# Set up error handling
set -e
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# Configuration
PORT=8501
LOG_FILE="server.log"
VENV_NAME="chatbot_env"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python() {
    if ! command_exists python3; then
        log "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l) )); then
        log "Python version $PYTHON_VERSION is too old. Please install Python 3.8 or higher."
        exit 1
    fi
    log "Python version $PYTHON_VERSION is compatible."
}

# Create and activate virtual environment
setup_venv() {
    if [ ! -d "$VENV_NAME" ]; then
        log "Creating virtual environment..."
        python3 -m venv $VENV_NAME
    fi
    
    log "Activating virtual environment..."
    source $VENV_NAME/bin/activate
}

# Install dependencies
install_dependencies() {
    log "Installing required packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install tensorflow numpy nltk streamlit
}

# Download NLTK data
download_nltk_data() {
    log "Downloading NLTK data..."
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
}

# Train model if needed
train_model() {
    if [ ! -f "chatbot_model.h5" ] || [ ! -f "words.pkl" ] || [ ! -f "classes.pkl" ]; then
        log "Training model..."
        rm -f chatbot_model.h5 words.pkl classes.pkl
        python3 train_model.py
    else
        log "Model files already exist. Skipping training."
    fi
}

# Start the server
start_server() {
    log "Starting Streamlit server on port $PORT..."
    streamlit run app.py \
        --server.port=$PORT \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --browser.serverAddress=0.0.0.0 \
        --browser.serverPort=$PORT
}

# Main execution
main() {
    log "Starting server setup..."
    
    # Check Python
    check_python
    
    # Setup virtual environment
    setup_venv
    
    # Install dependencies
    install_dependencies
    
    # Download NLTK data
    download_nltk_data
    
    # Train model if needed
    train_model
    
    # Start server
    start_server
}

# Run main function
main

deactivate 