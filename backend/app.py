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
CORS(app)  # Allow React to communicate with this backend

# --- LOAD RESOURCES ---
print("Loading model and resources...")
model = tf.keras.models.load_model("sentiment_model.h5")

with open("word_index.json", "r") as f:
    word_index = json.load(f)

class_names = np.load("classes.npy", allow_pickle=True)

# --- REUSE YOUR PREPROCESSING LOGIC ---
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

# Preprocessing helper matching your training
def encode_text(text):
    # 1. Clean
    cleaned_text = nlp_clean(text)
    # 2. Tokenize (simple split as used in your simple_preprocess logic)
    tokens = cleaned_text.split() 
    # 3. Map to integers
    sequence = [word_index[word] for word in tokens if word in word_index]
    # 4. Pad (MAX_LEN was 200 in your code)
    padded = pad_sequences([sequence], maxlen=200, padding="post")
    return padded
@app.route("/", methods=["POST"])
def predict():
    data = request.json
    user_text = data.get('text', '')
    
    if not user_text:
        return jsonify({'error': 'No text provided'}), 400

    # Process and Predict
    processed_input = encode_text(user_text)
    prediction_probs = model.predict(processed_input)
    predicted_index = np.argmax(prediction_probs)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction_probs))

    return jsonify({
        'class': predicted_class,
        'confidence': f"{confidence:.2%}"
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)