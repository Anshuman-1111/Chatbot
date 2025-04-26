import os
import sys
import subprocess
import platform
import pkg_resources

def check_python_version():
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python version must be 3.8 or higher")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    print("\nChecking dependencies...")
    required_packages = {
        'streamlit': '1.32.0',
        'tensorflow': '2.13.0',
        'numpy': '1.24.3',
        'nltk': '3.8.1',
        'scikit-learn': '1.3.0',
        'pillow': '10.2.0',
        'pandas': '2.1.4',
        'plotly': '5.18.0',
        'gunicorn': '21.2.0',
        'python-dotenv': '1.0.0'
    }
    
    missing_packages = []
    for package, version in required_packages.items():
        try:
            pkg_resources.require(f"{package}=={version}")
            print(f"✅ {package} {version} is installed")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing_packages.append(f"{package}=={version}")
    
    if missing_packages:
        print("\n❌ Missing packages:")
        for package in missing_packages:
            print(f"- {package}")
        return False
    return True

def check_model_files():
    print("\nChecking model files...")
    required_files = ['chatbot_model.h5', 'words.pkl', 'classes.pkl', 'intents.json']
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            missing_files.append(file)
            print(f"❌ {file} is missing")
    
    if missing_files:
        print("\nMissing model files. Please ensure all required files are present.")
        return False
    return True

def check_streamlit_config():
    print("\nChecking Streamlit configuration...")
    config_path = '.streamlit/config.toml'
    if os.path.exists(config_path):
        print(f"✅ Streamlit config exists at {config_path}")
        return True
    else:
        print(f"❌ Streamlit config is missing at {config_path}")
        return False

def check_git_setup():
    print("\nChecking Git setup...")
    try:
        subprocess.run(['git', 'status'], check=True, capture_output=True)
        print("✅ Git repository is initialized")
        return True
    except subprocess.CalledProcessError:
        print("❌ Git repository is not initialized")
        return False

def main():
    print("Starting deployment troubleshooting...\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("Streamlit Config", check_streamlit_config),
        ("Git Setup", check_git_setup)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n=== {check_name} ===")
        if not check_func():
            all_passed = False
    
    if all_passed:
        print("\n✅ All checks passed! Your environment is ready for deployment.")
    else:
        print("\n❌ Some checks failed. Please fix the issues before deploying.")
        print("\nCommon Solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Ensure all model files are in the correct directory")
        print("3. Initialize Git repository: git init")
        print("4. Create Streamlit config: mkdir -p .streamlit && touch .streamlit/config.toml")
        print("5. Check Python version: python --version")

if __name__ == "__main__":
    main() 