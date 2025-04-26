#!/bin/bash

# Set up logging
LOG_FILE="server.log"
echo "Starting server setup at $(date)" > $LOG_FILE

# Check Python version
echo "Checking Python version..." | tee -a $LOG_FILE
python --version >> $LOG_FILE 2>&1

# Install required packages
echo "Installing required packages..." | tee -a $LOG_FILE
pip install -r requirements.txt >> $LOG_FILE 2>&1

# Download NLTK data
echo "Downloading NLTK data..." | tee -a $LOG_FILE
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')" >> $LOG_FILE 2>&1

# Train the model if not already trained
if [ ! -f "chatbot_model.h5" ] || [ ! -f "words.pkl" ] || [ ! -f "classes.pkl" ]; then
    echo "Training model..." | tee -a $LOG_FILE
    python train_model.py >> $LOG_FILE 2>&1
fi

# Start the Streamlit server
echo "Starting Streamlit server..." | tee -a $LOG_FILE
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 >> $LOG_FILE 2>&1 