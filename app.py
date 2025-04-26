import streamlit as st
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {e}")
    st.error("Error initializing the chatbot. Please try again later.")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data with error handling
def load_model_and_data():
    try:
        with open('intents.json') as file:
            intents = json.load(file)
        logger.info("Intents loaded successfully")
        
        model = tf.keras.models.load_model('chatbot_model.h5')
        logger.info("Model loaded successfully")
        
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        logger.info("Vocabulary and classes loaded successfully")
        
        return intents, model, words, classes
    except Exception as e:
        logger.error(f"Error loading model or data: {e}")
        st.error("Error loading the chatbot model. Please check the logs for details.")
        return None, None, None, None

# Load the model and data
intents, model, words, classes = load_model_and_data()

def clean_up_sentence(sentence):
    try:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    except Exception as e:
        logger.error(f"Error cleaning sentence: {e}")
        return []

def bow(sentence, words):
    try:
        sentence_words = clean_up_sentence(sentence)
        bag = [1 if w in sentence_words else 0 for w in words]
        return np.array(bag)
    except Exception as e:
        logger.error(f"Error creating bag of words: {e}")
        return np.zeros(len(words))

def predict_class(sentence):
    try:
        bow_vector = bow(sentence, words)
        bow_vector = np.expand_dims(bow_vector, axis=0)
        res = model.predict(bow_vector)[0]
        threshold = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    except Exception as e:
        logger.error(f"Error predicting class: {e}")
        return []

def get_response(intents_list, intents_json):
    try:
        if not intents_list:
            return "Sorry, I didn't understand that. Can you rephrase?"
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except Exception as e:
        logger.error(f"Error getting response: {e}")
        return "I'm having trouble processing your request. Please try again."

# Set page config
st.set_page_config(
    page_title="Shopping Chatbot",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'price_range' not in st.session_state:
    st.session_state.price_range = (0, 2000)

# Custom CSS for better UI
st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .stButton>button {
        width: 100%;
    }
    .category-button {
        margin: 0.5rem;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🛍️ Shopping Assistant")
st.markdown("""
    Welcome to your personal shopping assistant! I can help you with:
    - Finding products across various categories
    - Checking prices and availability
    - Getting recommendations
    - Learning about discounts and offers
    - Understanding payment and delivery options
""")

# Sidebar for filters and categories
with st.sidebar:
    st.title("🛍️ Filters")
    
    # Price range slider
    st.subheader("Price Range")
    price_range = st.slider(
        "Select price range ($)",
        min_value=0,
        max_value=2000,
        value=(0, 2000),
        step=50
    )
    st.session_state.price_range = price_range
    
    # Category selection
    st.subheader("Categories")
    categories = ["Electronics", "Clothing", "Home Appliances", "Sports"]
    for category in categories:
        if st.button(category, key=f"cat_{category}"):
            st.session_state.selected_category = category
            st.session_state.chat_history.append({
                "role": "user",
                "content": f"Show me {category.lower()}"
            })

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Chat container
    chat_container = st.container()

    # Chat input
    with st.form("chat_form"):
        user_input = st.text_input("Type your message here...", key="user_input")
        col1, col2 = st.columns([1, 1])
        with col1:
            send_button = st.form_submit_button("Send")
        with col2:
            clear_button = st.form_submit_button("Clear Chat")

    # Process user input
    if send_button and user_input:
        try:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            intents_result = predict_class(user_input)
            response = get_response(intents_result, intents)
            st.session_state.chat_history.append({"role": "bot", "content": response})
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            st.error("An error occurred while processing your message. Please try again.")

    # Clear chat if button pressed
    if clear_button:
        st.session_state.chat_history = []
        st.experimental_rerun()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user">👤 You: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot">🤖 Bot: {message["content"]}</div>', unsafe_allow_html=True)

with col2:
    st.title("📊 Quick Actions")
    
    # Quick action buttons
    if st.button("🛒 View Cart"):
        st.session_state.chat_history.append({
            "role": "user",
            "content": "Show me my cart"
        })
    
    if st.button("💳 Payment Options"):
        st.session_state.chat_history.append({
            "role": "user",
            "content": "What payment methods do you accept"
        })
    
    if st.button("🚚 Delivery Info"):
        st.session_state.chat_history.append({
            "role": "user",
            "content": "What are the delivery options"
        })
    
    if st.button("🎁 Special Offers"):
        st.session_state.chat_history.append({
            "role": "user",
            "content": "Show me current discounts"
        })
    
    # Display selected category and price range
    st.subheader("Current Filters")
    if st.session_state.selected_category:
        st.write(f"Category: {st.session_state.selected_category}")
    st.write(f"Price Range: ${st.session_state.price_range[0]} - ${st.session_state.price_range[1]}")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>© 2024 Shopping Assistant. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)