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

app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
CORS(app)

# --- MODEL AND RESOURCES ---
print("Loading model and resources...")
model = tf.keras.models.load_model("sentiment_model.h5")

with open("word_index.json", "r") as f:
    word_index = json.load(f)

class_names = np.load("classes.npy", allow_pickle=True)

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

# Catch all route to serve React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
