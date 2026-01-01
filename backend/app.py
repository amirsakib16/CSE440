from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import os

# -----------------------------
# 1️⃣ Flask App Setup
# -----------------------------
# Absolute path to frontend build folder
frontend_build_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend/build')

app = Flask(__name__, static_folder=frontend_build_path, static_url_path='/')
CORS(app)  # Allow frontend to call API

# -----------------------------
# 2️⃣ Load Model & Resources
# -----------------------------
print("Loading model and resources...")
model = tf.keras.models.load_model("sentiment_model.h5")

with open("word_index.json", "r") as f:
    word_index = json.load(f)

class_names = np.load("classes.npy", allow_pickle=True)

# Download stopwords (ensure done once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# 3️⃣ Preprocessing Functions
# -----------------------------
def nlp_clean(text):
    """Clean and preprocess text for the model"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML
    text = text.replace('\\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    return " ".join(cleaned_words)

def encode_text(text):
    """Convert text to padded sequence for model"""
    cleaned_text = nlp_clean(text)
    tokens = cleaned_text.split()
    sequence = [word_index[word] for word in tokens if word in word_index]
    padded = pad_sequences([sequence], maxlen=200, padding="post")
    return padded

# -----------------------------
# 4️⃣ API Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({'error': 'No text provided'}), 400

    processed_input = encode_text(user_text)
    prediction_probs = model.predict(processed_input)
    predicted_index = np.argmax(prediction_probs)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction_probs))

    return jsonify({
        'class': predicted_class,
        'confidence': f"{confidence:.2%}"
    })

# -----------------------------
# 5️⃣ Serve React Frontend
# -----------------------------
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    file_path = os.path.join(app.static_folder, path)
    if path != "" and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        # always return index.html for React router
        return send_from_directory(app.static_folder, 'index.html')

# -----------------------------
# 6️⃣ Run Flask App
# -----------------------------
if __name__ == '__main__':
    # On Render, use host='0.0.0.0' instead of default
    app.run(debug=True, port=5000, host='0.0.0.0')
