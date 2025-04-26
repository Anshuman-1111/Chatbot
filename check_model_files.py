import os
import sys

required_files = ['chatbot_model.h5', 'words.pkl', 'classes.pkl', 'intents.json']
missing_files = []

print("Checking for required model files...")
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file} found")
    else:
        print(f"❌ {file} missing")
        missing_files.append(file)

if missing_files:
    print("\nError: The following required files are missing:")
    for file in missing_files:
        print(f"- {file}")
    print("\nPlease ensure all model files are present in the repository.")
    sys.exit(1)
else:
    print("\n✅ All required model files are present.")
    sys.exit(0) 