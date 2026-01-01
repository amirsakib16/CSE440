from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

# Initialize App
app = Flask(__name__)
CORS(app)

# --- LOAD RESOURCES ---
print("Loading model and resources...")
model = tf.keras.models.load_model("sentiment_model.h5")

with open("word_index.json", "r") as f:
    word_index = json.load(f)

class_names = np.load("classes.npy", allow_pickle=True)

# --- NLP PREPROCESSING ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def nlp_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def encode_text(text):
    cleaned_text = nlp_clean(text)
    tokens = cleaned_text.split() 
    sequence = [word_index[word] for word in tokens if word in word_index]
    padded = pad_sequences([sequence], maxlen=200, padding="post")
    return padded

# --- COMBINED ROUTE ---

@app.route("/", methods=["GET", "POST"], strict_slashes=False)
def handle_root():
    # 1. Handle GET (Browser visits, Render health checks)
    if request.method == "GET":
        return jsonify({
            "status": "Online",
            "message": "NLP Server is active. Send a POST request with {'text': '...'} to this same URL to get a prediction."
        }), 200

    # 2. Handle POST (React App requests)
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        user_text = data.get('text', '')
        
        # Process and Predict
        processed_input = encode_text(user_text)
        prediction_probs = model.predict(processed_input)
        predicted_index = np.argmax(prediction_probs)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction_probs))

        return jsonify({
            'class': str(predicted_class),
            'confidence': f"{confidence:.2%}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Silence favicon 404s
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, port=5000)