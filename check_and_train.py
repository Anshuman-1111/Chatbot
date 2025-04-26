import os
import logging
import sys
from train_model import main as train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_model_files():
    """Check if all required model files exist."""
    required_files = ['chatbot_model.h5', 'words.pkl', 'classes.pkl', 'intents.json']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    logger.info("All required model files exist.")
    return True

def main():
    """Main function to check and train the model if needed."""
    try:
        # Check if model files exist
        if not check_model_files():
            logger.info("Training new model...")
            # Train the model
            if train_model() != 0:
                logger.error("Model training failed.")
                return 1
        
        logger.info("Model check completed successfully.")
        return 0
    
    except Exception as e:
        logger.error(f"Error during model check: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 