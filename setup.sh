#!/bin/bash

# Exit on error
set -e

echo "Starting setup process..."

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip with retry mechanism
echo "Upgrading pip..."
for i in {1..3}; do
    if python -m pip install --upgrade pip; then
        echo "Pip upgraded successfully"
        break
    else
        echo "Attempt $i failed, retrying..."
        if [ $i -eq 3 ]; then
            echo "Failed to upgrade pip after 3 attempts"
            exit 1
        fi
        sleep 2
    fi
done

# Install NLTK data
echo "Installing NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')" || {
    echo "Failed to download NLTK data"
    exit 1
}

# Install requirements with retry mechanism
echo "Installing requirements..."
for i in {1..3}; do
    if python -m pip install --no-cache-dir -r requirements.txt; then
        echo "Requirements installed successfully"
        break
    else
        echo "Attempt $i failed, retrying..."
        if [ $i -eq 3 ]; then
            echo "Failed to install requirements after 3 attempts"
            exit 1
        fi
        sleep 5
    fi
done

# Verify installations
echo "Verifying installations..."
python -c "import streamlit; import tensorflow; import nltk; import numpy; import sklearn; import pandas; import plotly" || {
    echo "Failed to verify installations"
    exit 1
}

# Create necessary directories
echo "Creating required directories..."
mkdir -p .streamlit
mkdir -p models

# Set proper permissions
chmod -R 755 .

echo "Setup completed successfully!" 