import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import random
import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_nltk_data():
    try:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('wordnet')
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        raise

def load_intents():
    try:
        logger.info("Loading intents from intents.json...")
        with open('intents.json') as file:
            intents = json.load(file)
        logger.info("Intents loaded successfully")
        return intents
    except Exception as e:
        logger.error(f"Error loading intents: {e}")
        raise

def create_training_data(intents):
    try:
        logger.info("Creating training data...")
        words = []
        classes = []
        documents = []
        ignore_letters = ['?', '!', '.', ',']

        for intent in intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                words.extend(word_list)
                documents.append((word_list, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))

        training = []
        output_empty = [0] * len(classes)

        for document in documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            
            for word in words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            output_row = list(output_empty)
            output_row[classes.index(document[1])] = 1
            
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        logger.info("Training data created successfully")
        return words, classes, train_x, train_y
    except Exception as e:
        logger.error(f"Error creating training data: {e}")
        raise

def create_model(train_x, train_y):
    try:
        logger.info("Creating model...")
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        logger.info("Model created successfully")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise

def train_model(model, train_x, train_y):
    try:
        logger.info("Training model...")
        model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        logger.info("Model trained successfully")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def save_model_and_data(model, words, classes):
    try:
        logger.info("Saving model and data...")
        
        # Create backup of existing files if they exist
        if os.path.exists('chatbot_model.h5'):
            os.rename('chatbot_model.h5', 'chatbot_model.h5.bak')
        if os.path.exists('words.pkl'):
            os.rename('words.pkl', 'words.pkl.bak')
        if os.path.exists('classes.pkl'):
            os.rename('classes.pkl', 'classes.pkl.bak')
        
        # Save new files
        try:
            model.save('chatbot_model.h5', save_format='h5')
            with open('words.pkl', 'wb') as f:
                pickle.dump(words, f)
            with open('classes.pkl', 'wb') as f:
                pickle.dump(classes, f)
            
            # Remove backups if save was successful
            if os.path.exists('chatbot_model.h5.bak'):
                os.remove('chatbot_model.h5.bak')
            if os.path.exists('words.pkl.bak'):
                os.remove('words.pkl.bak')
            if os.path.exists('classes.pkl.bak'):
                os.remove('classes.pkl.bak')
            
            logger.info("Model and data saved successfully")
        except Exception as e:
            # Restore backups if save failed
            logger.error(f"Error saving files: {e}")
            if os.path.exists('chatbot_model.h5.bak'):
                os.rename('chatbot_model.h5.bak', 'chatbot_model.h5')
            if os.path.exists('words.pkl.bak'):
                os.rename('words.pkl.bak', 'words.pkl')
            if os.path.exists('classes.pkl.bak'):
                os.rename('classes.pkl.bak', 'classes.pkl')
            raise
    except Exception as e:
        logger.error(f"Error in save_model_and_data: {e}")
        raise

def main():
    try:
        # Check if intents.json exists
        if not os.path.exists('intents.json'):
            raise FileNotFoundError("intents.json not found. Please create it first.")

        # Download NLTK data
        download_nltk_data()

        # Load intents
        intents = load_intents()

        # Create training data
        words, classes, train_x, train_y = create_training_data(intents)

        # Create and train model
        model = create_model(train_x, train_y)
        train_model(model, train_x, train_y)

        # Save model and data
        save_model_and_data(model, words, classes)

        logger.info("Model training completed successfully!")
        print("\n✅ Model training completed successfully!")
        print("Generated files:")
        print("- chatbot_model.h5")
        print("- words.pkl")
        print("- classes.pkl")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main()) 