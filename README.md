# Shopping Assistant Chatbot

A Streamlit-based shopping assistant chatbot that helps users with product inquiries, price checks, and shopping-related questions.

## Features

- Product search and recommendations
- Price range filtering
- Category-based browsing
- Delivery information
- Payment options
- Special offers and discounts

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd shopping-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train_model.py
```

5. Run the application:
```bash
streamlit run app.py
```

## Deployment

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository
5. Set the main file path to `app.py`
6. Click "Deploy"

## Required Files

- `app.py` - Main application file
- `intents.json` - Chatbot intents and responses
- `chatbot_model.h5` - Trained model
- `words.pkl` - Vocabulary
- `classes.pkl` - Intent classes
- `requirements.txt` - Dependencies
- `Procfile` - Deployment configuration
- `runtime.txt` - Python version

## Project Structure

```
shopping-chatbot/
├── app.py              # Main application
├── intents.json        # Chatbot intents and responses
├── chatbot_model.h5    # Trained model
├── words.pkl           # Vocabulary
├── classes.pkl         # Intent classes
├── train_model.py      # Model training script
├── requirements.txt    # Dependencies
├── Procfile           # Deployment configuration
├── runtime.txt        # Python version
└── README.md          # Documentation
```

## Troubleshooting

If you encounter any issues:

1. Check if all required files are present
2. Verify Python version compatibility
3. Ensure all dependencies are installed
4. Check the Streamlit Cloud logs for errors

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License. 