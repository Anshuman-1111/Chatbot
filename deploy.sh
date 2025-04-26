#!/bin/bash

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install Git first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files to git
echo "Adding files to git..."
git add .

# Commit changes
echo "Committing changes..."
git commit -m "Initial commit for deployment"

# Instructions for Streamlit Cloud deployment
echo "
Deployment Instructions:
1. Create a GitHub repository if you haven't already
2. Add the remote repository:
   git remote add origin <your-github-repo-url>
3. Push your code:
   git push -u origin main
4. Go to https://streamlit.io/cloud
5. Sign in with your GitHub account
6. Click 'New app'
7. Select your repository and branch
8. Set the main file path to 'app.py'
9. Click 'Deploy'

Your app will be available at: https://<your-app-name>.streamlit.app
"

# Instructions for Heroku deployment
echo "
For Heroku deployment:
1. Install Heroku CLI
2. Run:
   heroku login
   heroku create
   git push heroku main
"

# Instructions for running the application locally
echo "
To run the application locally:
1. Install dependencies:
   pip install -r requirements.txt
2. Start the Streamlit server:
   streamlit run app.py
3. Open your browser and navigate to:
   http://localhost:8501
" 