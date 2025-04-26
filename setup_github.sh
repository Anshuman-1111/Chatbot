#!/bin/bash

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI is not installed. Please install it first:"
    echo "Windows: winget install GitHub.cli"
    echo "Mac: brew install gh"
    echo "Linux: sudo apt install gh"
    exit 1
fi

# Check if user is logged in to GitHub
if ! gh auth status &> /dev/null; then
    echo "Please log in to GitHub first:"
    echo "gh auth login"
    exit 1
fi

# Get repository name
read -p "Enter your GitHub repository name: " repo_name
read -p "Enter repository description: " repo_desc

# Create repository
echo "Creating GitHub repository..."
gh repo create "$repo_name" --public --description "$repo_desc"

# Initialize local repository if not already done
if [ ! -d ".git" ]; then
    git init
fi

# Add remote
git remote add origin "https://github.com/$(gh api user | jq -r .login)/$repo_name.git"

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Shopping Assistant Chatbot"

# Push to GitHub
git push -u origin main

echo "
GitHub repository setup complete!
Repository URL: https://github.com/$(gh api user | jq -r .login)/$repo_name
" 